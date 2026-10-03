import importlib

from backend.analyzers.text.analyzer import analyze_text
from backend.reasoning.crossmodal import run_crossmodal
from backend.reasoning.fusion import fuse
from backend.reasoning.graph import build_graph

# ML-A owns these. Missing analyzers are skipped so the pipeline never crashes.
ANALYZERS = {
    "image": ("backend.analyzers.image.analyzer", "analyze_image"),
    "video": ("backend.analyzers.video.analyzer", "analyze_video"),
    "audio": ("backend.analyzers.audio.analyzer", "analyze_audio"),
}


def _load(module_path: str, func_name: str):
    try:
        return getattr(importlib.import_module(module_path), func_name)
    except Exception:
        return None

def _normalize(name: str, result):
    """Make analyzer output match the frontend's field names."""
    if name == "video" and isinstance(result, dict):
        if result.get("manipulation_prob") is None and result.get("synthetic_prob") is not None:
            result["manipulation_prob"] = result["synthetic_prob"]
    return result

def run_pipeline(case: dict) -> dict:
    """case = {"image": path|None, "video": path|None, "audio": path|None, "claim": str|None}"""
    results, notes = {}, []

    for name, (module_path, func_name) in ANALYZERS.items():
        path = case.get(name)
        if not path:
            continue
        analyzer = _load(module_path, func_name)
        if analyzer is None:
            notes.append(f"{name} analyzer not available yet")
            continue
        try:
             results[name] = _normalize(name, analyzer(path))
        except Exception as exc:
            notes.append(f"{name} analysis failed: {exc}")

    text_result = analyze_text(case.get("claim")) if case.get("claim") else None

    findings = run_crossmodal(case, results, text_result)
    fused = fuse(results, findings)
    graph = build_graph(fused["risks"], findings)

    modalities = dict(results)
    if text_result:
        modalities["text"] = text_result

    return {
        "verdict": fused["verdict"],
        "leaning": fused["leaning"],
        "trust_score": fused["trust_score"],
        "confidence": fused["confidence"],
        "abstain_reason": fused["abstain_reason"],
        "modalities": modalities,
        "cross_modal": findings,
        "evidence_graph": graph,
        "explanation": fused["explanation"],
        "notes": notes,
    }