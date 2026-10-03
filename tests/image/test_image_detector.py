"""Detector tests.

* Plumbing tests use a tiny randomly initialised ViT built locally. They verify
  loading, preprocessing, the logit -> probability maths and caching. The
  predictions themselves carry no authenticity meaning.
* The integration test uses the real pretrained model and only runs when it is
  already in the local Hugging Face cache (or downloading is explicitly allowed
  with TRUSTLAYER_ALLOW_MODEL_DOWNLOAD=1). It never downloads on its own.
"""

from __future__ import annotations

import math
import os

import pytest
from PIL import Image

from backend.analyzers.image import ImageAnalyzer, ImageAnalyzerConfig
from backend.analyzers.image.detector import (
    DEFAULT_MODEL_ID,
    CommunityForensicsDetector,
    get_detector,
    reset_detector_cache,
)
from backend.analyzers.image.errors import ModelLoadError
from image_test_helpers import make_pixels

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_probabilities_follow_logit(tiny_model_dir):
    detector = CommunityForensicsDetector(str(tiny_model_dir))
    result = detector.predict(make_pixels())
    logit = result["raw_logit"]
    assert result["ai_generated_probability"] == pytest.approx(1 / (1 + math.exp(-logit)), rel=1e-5)
    assert result["ai_generated_probability"] + result["real_probability"] == pytest.approx(1.0, abs=1e-6)
    expected = "ai_generated" if result["ai_generated_probability"] >= 0.5 else "real"
    assert result["label"] == expected
    assert result["probability_calibrated"] is False


def test_prediction_is_deterministic_and_batch_consistent(tiny_model_dir):
    detector = CommunityForensicsDetector(str(tiny_model_dir))
    a, b = make_pixels(seed=1), make_pixels(seed=2)
    single_a, single_b = detector.predict(a), detector.predict(b)
    assert detector.predict(a)["raw_logit"] == single_a["raw_logit"]
    batch = detector.predict_batch([a, b])
    assert batch[0]["raw_logit"] == pytest.approx(single_a["raw_logit"], abs=1e-5)
    assert batch[1]["raw_logit"] == pytest.approx(single_b["raw_logit"], abs=1e-5)
    assert single_a["raw_logit"] != single_b["raw_logit"]


def test_model_loaded_once_and_registry_shared(tiny_model_dir):
    reset_detector_cache()
    first = get_detector(str(tiny_model_dir))
    second = get_detector(str(tiny_model_dir))
    assert first is second
    first.load()
    model = first._model
    first.predict(make_pixels())
    first.load()
    assert first._model is model


def test_device_selection_and_model_info(tiny_model_dir):
    detector = CommunityForensicsDetector(str(tiny_model_dir))
    info = detector.model_info()
    assert info["device"] == ("cuda" if torch.cuda.is_available() else "cpu")
    assert info["num_labels"] == 1 and info["input_size"] == 32
    assert info["architecture"] == "ViTForImageClassification"


def test_rejects_multi_label_head(tmp_path):
    from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

    config = ViTConfig(image_size=32, patch_size=16, hidden_size=32, num_hidden_layers=1,
                       num_attention_heads=2, intermediate_size=64, num_labels=2)
    ViTForImageClassification(config).save_pretrained(tmp_path)
    ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(tmp_path)
    with pytest.raises(ModelLoadError, match="single-logit"):
        CommunityForensicsDetector(str(tmp_path)).load()


def test_exif_orientation_applied_before_inference(tiny_model_dir, tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6
    path = tmp_path / "rot.jpg"
    make_pixels(160, 120).save(path, exif=exif)
    with Image.open(path) as image:
        image.load()
        result = CommunityForensicsDetector(str(tiny_model_dir)).predict(image)
    assert result["preprocessing"]["exif_transposed"] is True


def test_analyzer_output_schema_with_model(tiny_model_dir, jpeg_path):
    config = ImageAnalyzerConfig(model_id=str(tiny_model_dir))
    result = ImageAnalyzer(config).analyze(jpeg_path)
    assert result["status"] == "success", result["errors"]
    prediction = result["prediction"]
    assert {"label", "ai_generated_probability", "real_probability", "raw_logit", "model"} <= set(prediction)
    assert result["model_info"]["name"] == str(tiny_model_dir)
    assert result["uncertainty"]["status"] in ("low", "medium", "high")
    assert result["uncertainty"]["probability_calibrated"] is False
    ml = [e for e in result["evidence"] if e["type"] == "ml_detector"]
    assert len(ml) == 1 and ml[0]["value"] == prediction["ai_generated_probability"]


def test_uncertainty_rules():
    assess = ImageAnalyzer._assess_uncertainty
    near = {"ai_generated_probability": 0.55, "decision_threshold": 0.5}
    far = {"ai_generated_probability": 0.98, "decision_threshold": 0.5}
    mid = {"ai_generated_probability": 0.8, "decision_threshold": 0.5}
    assert assess(near, (500, 500), None, 384, True)["status"] == "high"
    assert assess(mid, (500, 500), None, 384, True)["status"] == "medium"
    assert assess(far, (500, 500), None, 384, True)["status"] == "low"
    small = assess(far, (100, 100), None, 384, True)
    assert small["status"] == "medium" and small["quality_factors"]
    jpeg = {"jpeg": {"estimated_quality": 40}}
    assert assess(far, (500, 500), jpeg, 384, True)["status"] == "medium"
    assert assess(None, (500, 500), None, 384, True)["status"] == "unavailable"


def _real_model_available() -> bool:
    if os.environ.get("TRUSTLAYER_ALLOW_MODEL_DOWNLOAD") == "1":
        return True
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(DEFAULT_MODEL_ID, "config.json")
        return isinstance(cached, str)
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _real_model_available(), reason="real model not in local HF cache")
def test_real_model_end_to_end(jpeg_path):
    local_only = os.environ.get("TRUSTLAYER_ALLOW_MODEL_DOWNLOAD") != "1"
    config = ImageAnalyzerConfig(local_files_only=local_only)
    analyzer = ImageAnalyzer(config)
    first = analyzer.analyze(jpeg_path)
    assert first["status"] == "success", first["errors"]
    prediction = first["prediction"]
    assert prediction["model"] == DEFAULT_MODEL_ID
    assert 0.0 <= prediction["ai_generated_probability"] <= 1.0
    assert prediction["ai_generated_probability"] + prediction["real_probability"] == pytest.approx(1.0, abs=1e-6)
    assert first["model_info"]["input_size"] == 384
    assert analyzer.analyze(jpeg_path)["prediction"]["raw_logit"] == pytest.approx(prediction["raw_logit"], abs=1e-4)
