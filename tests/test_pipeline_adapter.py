from backend.pipeline import _adapt_image_result


def test_adapt_adds_probability_signals_and_datetime():
    raw = {
        "prediction": {"ai_generated_probability": 0.9},
        "evidence": [
            {"severity": "high", "description": "Strong AI-generation score"},
            {"severity": "info", "description": "Camera make present"},
        ],
        "metadata": {"exif": {"DateTimeOriginal": "2026:07:12 10:00:00"}},
    }
    out = _adapt_image_result(raw)
    assert out["synthetic_prob"] == 0.9
    assert "Strong AI-generation score" in out["signals"]
    assert out["metadata"]["datetime"].startswith("2026")


def test_adapt_handles_failed_analysis():
    out = _adapt_image_result({"prediction": None, "evidence": [], "metadata": None})
    assert out["synthetic_prob"] is None
    assert out["signals"] == []