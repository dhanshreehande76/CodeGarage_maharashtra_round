"""
Evidence fusion: modality risks + cross-modal findings -> verdict, trust score,
confidence, and an Uncertain (abstain) decision. Transparent rules, no black box.
"""

RISK_T = 0.6                     # modality risk at or above this = "risky"
CONFLICT_T = 0.6                 # conflict at or above this = "strong conflict"
AMBIG_LOW, AMBIG_HIGH = 0.4, 0.6 # values in this band are borderline
ABSTAIN_BELOW = 0.5              # confidence below this -> uncertain

RISK_KEYS = {
    "image": ("synthetic_prob",),
    "video": ("manipulation_prob", "synthetic_prob"),  # frontend name first, old name accepted
    "audio": ("spoof_prob",),
}



def modality_risks(results: dict) -> dict:
    """Only modalities whose analyzer returned a real probability count as evidence."""
    risks = {}
    for name, keys in RISK_KEYS.items():
        res = results.get(name)
        if not res:
            continue
        for key in keys:
            if res.get(key) is not None:
                risks[name] = float(res[key])
                break
    return risks


def _explain(verdict, leaning, risks, findings, reason):
    parts = []
    for name, risk in risks.items():
        if risk >= RISK_T:
            parts.append(f"{name} shows strong synthetic indicators ({risk:.0%})")
    for f in findings:
        if 1.0 - f["score"] >= CONFLICT_T:
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

    max_risk = max(risks.values(), default=0.0)
    avg_risk = sum(risks.values()) / len(risks) if risks else 0.0
    mean_conflict = sum(conflicts) / len(conflicts) if conflicts else 0.0

    overall_risk = 0.5 * max_risk + 0.2 * avg_risk + 0.3 * mean_conflict
    trust_score = round(1.0 - overall_risk, 2)

    if n_risky >= 2 or (n_risky >= 1 and n_strong >= 2):
        verdict = "coordinated_synthetic"
    elif n_risky >= 1 or n_strong >= 1:
        verdict = "manipulated"
    else:
        verdict = "authentic"

    # Confidence: more independent evidence and fewer borderline values -> higher
    values = list(risks.values()) + conflicts
    ambiguous = [v for v in values if AMBIG_LOW <= v < AMBIG_HIGH]
    ambiguous_fraction = len(ambiguous) / len(values) if values else 1.0
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

    return {
        "verdict": verdict,
        "leaning": leaning,
        "trust_score": trust_score,
        "confidence": confidence,
        "abstain_reason": reason,
        "risks": risks,
        "explanation": _explain(verdict, leaning, risks, findings, reason),
    }