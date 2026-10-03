from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.analyzers.image.hashing import hamming_distance, perceptual_hash, sha256_file
from backend.analyzers.image.metadata import MetadataAnalyzer
from image_test_helpers import make_pixels


def analyze(path: Path, **kwargs):
    with Image.open(path) as image:
        image.load()
        return MetadataAnalyzer().analyze(path, image, **kwargs)


def codes(meta: dict) -> set[str]:
    return {f["code"] for f in meta["findings"]}


def test_sha256_matches_hashlib(jpeg_path: Path):
    assert sha256_file(jpeg_path) == hashlib.sha256(jpeg_path.read_bytes()).hexdigest()


def test_sha256_changes_with_content(tmp_path: Path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    make_pixels(seed=1).save(a)
    make_pixels(seed=2).save(b)
    assert sha256_file(a) != sha256_file(b)


def test_perceptual_hash_is_separate_and_stable():
    rng = np.random.default_rng(3)
    blocks = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    image = Image.fromarray(np.kron(blocks, np.ones((16, 16, 1), dtype=np.uint8)))
    first, second = perceptual_hash(image), perceptual_hash(image.copy())
    assert first == second
    assert len(first["hash"]) == 16
    assert hamming_distance(first["hash"], first["hash"]) == 0
    resized = image.resize((64, 64))
    assert hamming_distance(first["hash"], perceptual_hash(resized)["hash"]) <= 8


def test_missing_exif_is_absent_not_suspicious(jpeg_path: Path):
    meta = analyze(jpeg_path)
    assert meta["status"] == "absent"
    assert "exif_absent" in codes(meta)
    finding = next(f for f in meta["findings"] if f["code"] == "exif_absent")
    assert finding["category"] == "absent"
    assert finding["severity"] == "info"
    assert "not evidence of AI generation" in finding["description"]
    assert meta["exif"]["available"] is False


def test_png_without_exif_is_absent(png_path: Path):
    meta = analyze(png_path)
    assert meta["format"] == "PNG"
    assert meta["mime_type"] == "image/png"
    assert meta["status"] == "absent"


def test_bmp_exif_unavailable(tmp_path: Path):
    path = tmp_path / "x.bmp"
    make_pixels().save(path)
    meta = analyze(path)
    assert meta["status"] == "unavailable"
    assert "exif_unavailable" in codes(meta)


def test_exif_extraction(exif_jpeg_path: Path):
    meta = analyze(exif_jpeg_path, reference_time=datetime(2025, 1, 1))
    exif = meta["exif"]
    assert meta["status"] == "present"
    assert exif["available"] is True
    assert exif["camera_make"] == "TestCam"
    assert exif["camera_model"] == "Model X"
    assert exif["software"] == "TestFirmware 1.0"
    assert exif["datetime"] == "2024:05:01 10:00:00"
    assert exif["datetime_original"] == "2024:05:01 09:59:59"
    assert exif["datetime_digitized"] == "2024:05:01 09:59:59"
    assert exif["orientation"] == 1
    assert exif["gps"]["latitude"] == pytest.approx(18.52, abs=1e-3)
    assert exif["gps"]["longitude"] == pytest.approx(73.86, abs=1e-3)
    assert {"exif_present", "camera_info_present", "gps_present"} <= codes(meta)
    assert meta["width"] == 160 and meta["height"] == 120
    assert meta["sha256"] is None


def test_sha256_is_embedded_when_supplied(jpeg_path: Path):
    digest = sha256_file(jpeg_path)
    assert analyze(jpeg_path, sha256=digest)["sha256"] == digest


def test_extension_format_mismatch(tmp_path: Path):
    path = tmp_path / "actually_png.jpg"
    make_pixels().save(path, format="PNG")
    meta = analyze(path)
    assert meta["status"] == "inconsistent"
    assert {"extension_format_mismatch", "mime_extension_mismatch"} <= codes(meta)


def test_declared_mime_mismatch(jpeg_path: Path):
    meta = analyze(jpeg_path, declared_mime_type="image/png")
    assert "mime_declared_mismatch" in codes(meta)
    assert "mime_declared_mismatch" not in codes(analyze(jpeg_path, declared_mime_type="image/jpeg"))


@pytest.mark.parametrize(
    "raw, code",
    [
        ("2024-05-01 10:00:00", "timestamp_malformed"),
        ("0000:00:00 00:00:00", "timestamp_placeholder"),
        ("2024:13:45 10:00:00", "timestamp_invalid"),
    ],
)
def test_structurally_invalid_timestamps(tmp_path: Path, raw: str, code: str):
    exif = Image.Exif()
    exif[0x0132] = raw
    path = tmp_path / "ts.jpg"
    make_pixels().save(path, exif=exif)
    meta = analyze(path)
    assert code in codes(meta)
    assert meta["status"] == "inconsistent"


def test_future_timestamp_is_suspicious(tmp_path: Path):
    exif = Image.Exif()
    exif[0x0132] = "2031:01:01 00:00:00"
    path = tmp_path / "future.jpg"
    make_pixels().save(path, exif=exif)
    meta = analyze(path, reference_time=datetime(2026, 1, 1))
    assert "timestamp_in_future" in codes(meta)


def test_generator_software_tag_is_suspicious(tmp_path: Path):
    exif = Image.Exif()
    exif[0x0131] = "Stable Diffusion XL"
    path = tmp_path / "gen.jpg"
    make_pixels().save(path, exif=exif)
    meta = analyze(path)
    assert meta["status"] == "suspicious"
    assert "generator_software_tag" in codes(meta)


def test_png_generation_parameters_chunk(tmp_path: Path):
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text("parameters", "a cat\nSteps: 20, Sampler: Euler a, CFG scale: 7")
    path = tmp_path / "sd.png"
    make_pixels().save(path, pnginfo=info)
    meta = analyze(path)
    assert "generation_parameters_chunk" in codes(meta)
    assert meta["text_chunks"]["parameters"].startswith("a cat")


def test_does_not_invent_metadata(jpeg_path: Path):
    exif = analyze(jpeg_path)["exif"]
    assert exif["camera_make"] is None and exif["gps"] is None and exif["tags"] == {}
