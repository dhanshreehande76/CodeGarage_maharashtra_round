from backend.reasoning.fusion import fuse


def run(img=None, aud=None, findings=()):
    results = {}
    if img is not None:
        results["image"] = {"synthetic_prob": img}
    if aud is not None:
        results["audio"] = {"spoof_prob": aud}
    return fuse(results, list(findings))


def test_authentic():
    out = run(0.05, 0.10, [
        {"pair": "image-text", "score": 0.90, "finding": "Image matches claim"},
        {"pair": "audio-text", "score": 0.85, "finding": "Transcript matches claim"},
    ])
    assert out["verdict"] == "authentic"
    assert out["trust_score"] >= 0.8


def test_manipulated():
    out = run(0.90, None, [{"pair": "image-text", "score": 0.20, "finding": "Image contradicts claim"}])
    assert out["verdict"] == "manipulated"


def test_coordinated():
    out = run(0.91, 0.77, [
        {"pair": "image-text", "score": 0.21, "finding": "Image contradicts claim"},
        {"pair": "audio-text", "score": 0.34, "finding": "Transcript contradicts claim"},
    ])
    assert out["verdict"] == "coordinated_synthetic"
    assert out["trust_score"] < 0.3


def test_uncertain_borderline():
    out = run(0.52, None, [{"pair": "image-text", "score": 0.55, "finding": "Weak match"}])
    assert out["verdict"] == "uncertain"
    assert out["abstain_reason"]


def test_uncertain_single_source():
    out = run(0.05)
    assert out["verdict"] == "uncertain"
def test_image_real_analyzer_schema():
    results = {
        "image": {
            "prediction": {
                "ai_generated_probability": 0.90,
            }
        }
    }

    out = fuse(results, [])

    assert out["risks"]["image"] == 0.90
    assert out["verdict"] == "manipulated"

