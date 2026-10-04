"""
Evidence fusion: modality risks + cross-modal findings -> verdict, trust score,
confidence, and an Uncertain (abstain) decision. Transparent rules, no black box.
"""

RISK_T = 0.6                     # modality risk at or above this = "risky"
CONFLICT_T = 0.6                 # conflict at or above this = "strong conflict"
AMBIG_LOW, AMBIG_HIGH = 0.4, 0.6 # values in this band are borderline
ABSTAIN_BELOW = 0.5              # confidence below this -> uncertain
TRUST_CAP = 0.95                 # never claim certainty from automated checks

RISK_KEYS = {
    "image": ("synthetic_prob",),
    "video": ("manipulation_prob", "synthetic_prob"),
    "audio": ("spoof_prob",),
}


def modality_risks(results: dict) -> dict:
    """Only modalities whose analyzer returned a real probability count as evidence."""
    risks = {}
    for name, keys in RISK_KEYS.items():
        res = results.get(name)
        if not res:
            continue
        value = None
        for key in keys:
            if res.get(key) is not None:
                value = res[key]
                break
        # The real image analyzer nests its probability inside "prediction"
        if value is None and name == "image":
            value = (res.get("prediction") or {}).get("ai_generated_probability")
        if value is not None:
            risks[name] = float(value)
    return risks


def _explain(verdict, leaning, risks, findings, reason):
    parts = []
    risky = [name for name, risk in risks.items() if risk >= RISK_T]
    strong = [f for f in findings if 1.0 - f["score"] >= CONFLICT_T]

    if strong and risks and not risky:
        parts.append(
            "The media itself looks genuine, but it does not support the claim, "
            "which suggests out-of-context use"
        )
    for name, risk in risks.items():
        if risk >= RISK_T:
            parts.append(f"{name} shows strong synthetic indicators ({risk:.0%})")
    for f in strong:
        parts.append(f["finding"])

    if verdict == "uncertain":
        return f"{reason} Current leaning: {leaning.replace('_', ' ')}."
    if parts:
        return "; ".join(parts) + "."
    return "No strong manipulation indicators and the available evidence is consistent."


def fuse(results: dict, findings: list) -> dict:
    """
    results:  {"image": {"synthetic_prob": ...}, "audio": {"spoof_prob": ...}, ...}
    findings: [{"pair": "image-text", "score": 0-1 (1 = consistent), "finding": str}, ...]
    """
    risks = modality_risks(results)
    conflicts = [round(1.0 - f["score"], 4) for f in findings]

    n_risky = sum(r >= RISK_T for r in risks.values())
    n_strong = sum(c >= CONFLICT_T for c in conflicts)

    # Trust: dominated by the strongest warning sign, softened by the overall level
    signals = list(risks.values()) + conflicts
    if signals:
        peak = max(signals)
        mean_signal = sum(signals) / len(signals)
        overall_risk = 0.6 * peak + 0.4 * mean_signal
        trust_score = round(min(TRUST_CAP, 1.0 - overall_risk), 2)
    else:
        trust_score = 0.5  # no evidence at all: neutral, not "trusted"

    if n_risky >= 2 or (n_risky >= 1 and n_strong >= 2):
        verdict = "coordinated_synthetic"
    elif n_risky >= 1 or n_strong >= 1:
        verdict = "manipulated"
    else:
        verdict = "authentic"

    # Confidence: more independent evidence and fewer borderline values -> higher
    ambiguous = [v for v in signals if AMBIG_LOW <= v < AMBIG_HIGH]
    ambiguous_fraction = len(ambiguous) / len(signals) if signals else 1.0
    evidence = min(1.0, (len(risks) + len(findings)) / 4)
    confidence = round(min(0.97, 0.5 * evidence + 0.5 * (1.0 - ambiguous_fraction)), 2)

    # Abstain (Uncertain / Requires Verification)
    reason = None
    if not risks and not findings:
        reason = "No analyzable evidence was provided."
    elif verdict == "authentic" and not findings:
        reason = "Only a single source was checked, so consistency could not be verified."
    elif confidence < ABSTAIN_BELOW:
        reason = "Signals are weak or borderline, so the evidence is insufficient."

    leaning = verdict
    if reason:
        verdict = "uncertain"
        trust_score = min(trust_score, 0.5)  # an abstention is never shown as "high trust"  # an abstention is never shown as "high trust"

    return {
        "verdict": verdict,
        "leaning": leaning,
        "trust_score": trust_score,
        "confidence": confidence,
        "abstain_reason": reason,
        "risks": risks,
        "explanation": _explain(verdict, leaning, risks, findings, reason),
    }