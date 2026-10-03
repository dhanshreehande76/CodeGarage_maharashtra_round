"""
Evidence fusion: modality risks + cross-modal findings -> verdict, trust score,
confidence, and an Uncertain (abstain) decision. Transparent rules, no black box.
"""

RISK_T = 0.6
CONFLICT_T = 0.6
AMBIG_LOW, AMBIG_HIGH = 0.4, 0.6
ABSTAIN_BELOW = 0.5

RISK_KEYS = {
    "image": "synthetic_prob",
    "video": "synthetic_prob",
    "audio": "spoof_prob",
}


def modality_risks(results: dict) -> dict:
    """Extract modality risk probabilities while supporting analyzer schemas."""
    risks = {}

    for name, key in RISK_KEYS.items():
        res = results.get(name)

        if not res:
            continue

        value = res.get(key)

        # Current image analyzer stores its probability inside prediction.
        if value is None and name == "image":
            prediction = res.get("prediction") or {}
            value = prediction.get("ai_generated_probability")

        if value is not None:
            risks[name] = float(value)

    return risks


def _explain(verdict, leaning, risks, findings, reason):
    parts = []

    for name, risk in risks.items():
        if risk >= RISK_T:
            parts.append(
                f"{name} shows strong synthetic indicators ({risk:.0%})"
            )

    for f in findings:
        if 1.0 - f["score"] >= CONFLICT_T:
            parts.append(f"{f['finding']}")

    if verdict == "uncertain":
        return f"{reason} Current leaning: {leaning.replace('_', ' ')}."

    if parts:
        return "; ".join(parts) + "."

    return "No strong manipulation indicators and the available evidence is consistent."


def fuse(results: dict, findings: list) -> dict:
    """Fuse modality and cross-modal evidence."""

    risks = modality_risks(results)

    conflicts = [
        round(1.0 - f["score"], 4)
        for f in findings
    ]

    n_risky = sum(r >= RISK_T for r in risks.values())
    n_strong = sum(c >= CONFLICT_T for c in conflicts)

    max_risk = max(risks.values(), default=0.0)

    avg_risk = (
        sum(risks.values()) / len(risks)
        if risks
        else 0.0
    )

    mean_conflict = (
        sum(conflicts) / len(conflicts)
        if conflicts
        else 0.0
    )

    overall_risk = (
        0.5 * max_risk
        + 0.2 * avg_risk
        + 0.3 * mean_conflict
    )

    trust_score = round(1.0 - overall_risk, 2)

    if n_risky >= 2 or (n_risky >= 1 and n_strong >= 2):
        verdict = "coordinated_synthetic"
    elif n_risky >= 1 or n_strong >= 1:
        verdict = "manipulated"
    else:
        verdict = "authentic"

    values = list(risks.values()) + conflicts

    ambiguous = [
        v for v in values
        if AMBIG_LOW <= v < AMBIG_HIGH
    ]

    ambiguous_fraction = (
        len(ambiguous) / len(values)
        if values
        else 1.0
    )

    evidence = min(
        1.0,
        (len(risks) + len(findings)) / 4
    )

    confidence = round(
        min(
            0.97,
            0.5 * evidence
            + 0.5 * (1.0 - ambiguous_fraction)
        ),
        2,
    )

    reason = None

    if not risks:
        reason = "No analyzable evidence was provided."
    elif verdict == "authentic" and not findings:
        reason = (
            "Only a single source was checked, "
            "so consistency could not be verified."
        )
    elif confidence < ABSTAIN_BELOW:
        reason = (
            "Signals are weak or borderline, "
            "so the evidence is insufficient."
        )

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
        "explanation": _explain(
            verdict,
            leaning,
            risks,
            findings,
            reason,
        ),
    }
