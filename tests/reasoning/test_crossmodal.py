from backend.reasoning.crossmodal import (
    CONSISTENT_THRESHOLD,
    INCONSISTENT_THRESHOLD,
    _finding,
    run_crossmodal,
)


def test_finding_consistent():
    result = _finding(0.9, "image-text", "test")

    assert result["pair"] == "image-text"
    assert result["score"] == 0.9
    assert result["status"] == "consistent"


def test_finding_inconsistent():
    result = _finding(0.2, "image-text", "test")

    assert result["status"] == "inconsistent"


def test_finding_uncertain():
    score = (CONSISTENT_THRESHOLD + INCONSISTENT_THRESHOLD) / 2

    result = _finding(score, "image-text", "test")

    assert result["status"] == "uncertain"


def test_empty_case_returns_no_findings():
    result = run_crossmodal({}, {}, None)

    assert result == []


def test_claim_without_modalities_returns_no_findings():
    result = run_crossmodal(
        {"claim": "A person is standing outside."},
        {},
        {"claims": ["A person is standing outside."]},
    )

    assert result == []