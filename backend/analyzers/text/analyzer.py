def analyze_text(claim: str | None) -> dict:
    claim = (claim or "").strip()
    return {
        "claims": [claim] if claim else [],
        "entities": {"places": [], "dates": [], "people": []},
    }