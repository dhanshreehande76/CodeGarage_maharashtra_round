from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from backend.analyzers.image import ImageAnalyzer, ImageAnalyzerConfig
from backend.analyzers.image.errors import ModelLoadError
from backend.analyzers.image.schema import RESULT_KEYS, UNCERTAINTY_STATUSES
from image_test_helpers import make_pixels


@pytest.fixture
def analyzer() -> ImageAnalyzer:
    return ImageAnalyzer(ImageAnalyzerConfig(run_detector=False))


def assert_contract(result: dict) -> None:
    assert tuple(result.keys()) == RESULT_KEYS
    assert result["modality"] == "image"
    assert result["status"] in ("success", "partial", "error")
    assert set(result["forensics"]) == {
        "compression", "ela", "image_statistics", "frequency_analysis"
    }
    assert set(result["hashes"]) == {"sha256", "perceptual_hash"}
    assert result["uncertainty"]["status"] in UNCERTAINTY_STATUSES
    assert isinstance(result["evidence"], list)
    assert isinstance(result["warnings"], list) and isinstance(result["errors"], list)
    for item in result["evidence"]:
        assert set(item) == {"type", "severity", "description", "value"}
        assert item["severity"] in ("info", "low", "medium", "high")
    for error in result["errors"]:
        assert set(error) == {"code", "message", "stage"}
    json.dumps(result, allow_nan=False)


def test_valid_jpeg(analyzer, jpeg_path):
    result = analyzer.analyze(jpeg_path)
    assert_contract(result)
    assert result["status"] == "success"
    assert result["metadata"]["format"] == "JPEG"
    assert result["metadata"]["mime_type"] == "image/jpeg"
    assert (result["metadata"]["width"], result["metadata"]["height"]) == (160, 120)
    assert result["metadata"]["sha256"] == result["hashes"]["sha256"]
    assert result["forensics"]["ela"]["signal"]["signal"] == "ela_variation"
    assert result["forensics"]["compression"]["jpeg"]["estimated_quality"] == 90
    assert result["prediction"] is None
    assert result["uncertainty"]["status"] == "unavailable"


def test_valid_png(analyzer, png_path):
    result = analyzer.analyze(png_path)
    assert_contract(result)
    assert result["status"] == "success"
    assert result["metadata"]["format"] == "PNG"
    assert result["forensics"]["compression"]["jpeg"] is None
    assert result["forensics"]["ela"]["applicability"] == "limited"
    assert not any(e["type"] == "ela" for e in result["evidence"])


def test_missing_exif_is_info_evidence_only(analyzer, jpeg_path):
    result = analyzer.analyze(jpeg_path)
    exif_items = [e for e in result["evidence"] if e["value"] == "exif_absent"]
    assert exif_items and exif_items[0]["severity"] == "info"
    assert result["metadata"]["status"] == "absent"


def test_analysis_is_deterministic(analyzer, jpeg_path):
    assert analyzer.analyze(jpeg_path) == analyzer.analyze(jpeg_path)


def test_path_accepts_str(analyzer, jpeg_path):
    assert analyzer.analyze(str(jpeg_path))["status"] == "success"


def test_rgba_png_flattened_with_warning(analyzer, tmp_path):
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (96, 96), (255, 0, 0, 100)).save(path)
    result = analyzer.analyze(path)
    assert result["status"] == "success"
    assert any("Transparency" in w for w in result["warnings"])


def test_grayscale_and_palette_modes(analyzer, tmp_path):
    gray, pal = tmp_path / "g.png", tmp_path / "p.png"
    make_pixels().convert("L").save(gray)
    make_pixels().convert("P").save(pal)
    assert analyzer.analyze(gray)["status"] == "success"
    assert analyzer.analyze(pal)["status"] == "success"


def test_sixteen_bit_png(analyzer, tmp_path):
    import numpy as np

    path = tmp_path / "sixteen.png"
    Image.fromarray((np.arange(96 * 96, dtype=np.uint32).reshape(96, 96) * 7).astype("uint16")).save(path)
    result = analyzer.analyze(path)
    assert result["status"] == "success", result["errors"]


def test_nonexistent_file(analyzer, tmp_path):
    result = analyzer.analyze(tmp_path / "nope.jpg")
    assert_contract(result)
    assert result["status"] == "error"
    assert result["errors"][0]["code"] == "file_not_found"
    assert result["metadata"] is None


def test_corrupted_image_content(analyzer, tmp_path):
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"this is not a jpeg at all" * 20)
    result = analyzer.analyze(path)
    assert_contract(result)
    assert result["status"] == "error"
    assert result["errors"][0]["code"] == "corrupted_image"
    assert result["hashes"]["sha256"] is not None


def test_truncated_image(analyzer, jpeg_path, tmp_path):
    data = jpeg_path.read_bytes()
    path = tmp_path / "trunc.jpg"
    path.write_bytes(data[: len(data) // 2])
    result = analyzer.analyze(path)
    assert result["status"] == "error"
    assert result["errors"][0]["code"] == "corrupted_image"


def test_unsupported_file(analyzer, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    result = analyzer.analyze(path)
    assert_contract(result)
    assert result["status"] == "error"
    assert result["errors"][0]["code"] == "unsupported_file"


def test_unsupported_image_format(analyzer, tmp_path):
    path = tmp_path / "icon.ico"
    Image.new("RGB", (32, 32), "red").save(path)
    assert analyzer.analyze(path)["errors"][0]["code"] == "unsupported_file"


def test_empty_file_and_directory(analyzer, tmp_path):
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    assert analyzer.analyze(empty)["errors"][0]["code"] == "empty_file"
    assert analyzer.analyze(tmp_path)["errors"][0]["code"] == "not_a_file"


def test_huge_image_rejected_before_decode(jpeg_path, tmp_path):
    small_limit = ImageAnalyzer(ImageAnalyzerConfig(run_detector=False, max_pixels=1000))
    assert small_limit.analyze(jpeg_path)["errors"][0]["code"] == "image_too_large"
    tiny_file_limit = ImageAnalyzer(ImageAnalyzerConfig(run_detector=False, max_file_size_bytes=10))
    assert tiny_file_limit.analyze(jpeg_path)["errors"][0]["code"] == "file_too_large"


def test_model_load_failure_does_not_crash(jpeg_path, tmp_path):
    from backend.analyzers.image.detector import CommunityForensicsDetector

    missing = CommunityForensicsDetector(str(tmp_path / "no-such-model"), local_files_only=True)
    result = ImageAnalyzer(detector=missing).analyze(jpeg_path)
    assert_contract(result)
    assert result["status"] == "partial"
    assert result["errors"][0]["code"] == "model_load_failed"
    assert result["prediction"] is None
    assert result["uncertainty"]["status"] == "unavailable"
    assert result["forensics"]["ela"] is not None and result["metadata"] is not None


def test_inference_failure_does_not_crash(jpeg_path):
    class Exploding:
        def predict(self, image):
            from backend.analyzers.image.errors import InferenceError

            raise InferenceError("boom")

    result = ImageAnalyzer(detector=Exploding()).analyze(jpeg_path)
    assert result["status"] == "partial"
    assert result["errors"][0] == {"code": "inference_failed", "message": "boom", "stage": "detector"}


def test_model_load_error_is_cached_not_retried(tmp_path):
    from backend.analyzers.image.detector import CommunityForensicsDetector

    detector = CommunityForensicsDetector(str(tmp_path / "missing"), local_files_only=True)
    with pytest.raises(ModelLoadError) as first:
        detector.load()
    with pytest.raises(ModelLoadError) as second:
        detector.load()
    assert first.value is second.value
