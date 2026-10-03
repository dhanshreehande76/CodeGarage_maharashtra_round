from backend.reasoning.crossmodal import metadata_text_finding, run_crossmodal


def test_metadata_year_matches():
    out = metadata_text_finding({"datetime": "2026:07:12 10:11:12"}, "Flooding in Mumbai, July 2026")
    assert out["pair"] == "metadata-text"
    assert out["score"] >= 0.8


def test_metadata_year_mismatch():
    out = metadata_text_finding({"datetime": "2019:03:01 09:00:00"}, "Flooding happened in 2026")
    assert out["score"] <= 0.2


def test_metadata_missing_is_skipped():
    assert metadata_text_finding({}, "Flood in 2026") is None
    assert metadata_text_finding({"datetime": "2026:01:01 00:00:00"}, "no year here") is None


def test_no_claim_means_no_findings():
    assert run_crossmodal({"image": "x.jpg"}, {}, None) == []

from backend.reasoning.crossmodal import consistency_from_delta


def test_consistency_rises_with_delta():
    assert consistency_from_delta(-0.03) < consistency_from_delta(0.0) < consistency_from_delta(0.05)


def test_consistency_extremes():
    assert consistency_from_delta(0.12) > 0.9
    assert consistency_from_delta(-0.05) < 0.1