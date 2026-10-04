import importlib
import re

from backend.analyzers.text.analyzer import analyze_text
from backend.reasoning.crossmodal import run_crossmodal
from backend.reasoning.fusion import fuse
from backend.reasoning.graph import build_graph

# Audio and video analyzers are plain functions. Missing ones are skipped, with the reason in notes.
ANALYZERS = {
    "video": ("backend.analyzers.video.analyzer", "analyze_video"),
    "audio": ("backend.analyzers.audio.analyzer", "analyze_audio"),
}

_YEAR = re.compile(r"(?:19|20)\d{2}")
_DATE_KEY_HINTS = ("datetime", "date_time", "date_taken", "taken", "capture")


def _load(module_path: str, func_name: str):
    try:
        return getattr(importlib.import_module(module_path), func_name), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# ---------------- Image adapter (ImageAnalyzer is class-based) ----------------

_image_analyzer = None


def _find_datetime(node, depth: int = 0):
    """Find an EXIF-style capture date anywhere inside the metadata structure."""
    if depth > 4:
        return None
    if isinstance(node, dict):
        for key, value in node.items():
            name = str(key).lower()
            if isinstance(value, str) and _YEAR.search(value) and any(h in name for h in _DATE_KEY_HINTS):
                return value
        for value in node.values():
            found = _find_datetime(value, depth + 1)
            if found:
                return found
    elif isinstance(node, list):
        for item in node[:20]:
            found = _find_datetime(item, depth + 1)
            if found:
                return found
    return None


def _adapt_image_result(result: dict) -> dict:
    """Add the fields the fusion layer and the frontend expect, without removing anything."""
    prediction = result.get("prediction") or {}
    result["synthetic_prob"] = prediction.get("ai_generated_probability")

    evidence = result.get("evidence") or []
    strong = [e["description"] for e in evidence
              if e.get("severity") in ("medium", "high") and e.get("description")]
    other = [e["description"] for e in evidence
             if e.get("severity") not in ("medium", "high") and e.get("description")]
    result["signals"] = (strong + other[:3])[:6] if strong else other[:4]

    metadata = result.get("metadata")
    if isinstance(metadata, dict) and "datetime" not in metadata:
        found = _find_datetime(metadata)
        if found:
            metadata["datetime"] = found
    return result


def analyze_image(path: str) -> dict:
    global _image_analyzer
    if _image_analyzer is None:
        from backend.analyzers.image import ImageAnalyzer
        _image_analyzer = ImageAnalyzer()  # the detector model loads on the first prediction
    return _adapt_image_result(_image_analyzer.analyze(path))


# ---------------- Other normalisation ----------------

def _evidence_signals(result: dict, limit: int = 4) -> list:
    out = []
    for item in result.get("evidence") or []:
        description = item.get("description")
        if description and item.get("severity") in ("low", "medium", "high"):
            out.append(description)
    return out[:limit]


def _normalize(name: str, result):
    """Make analyzer output match the frontend's field names."""
    if not isinstance(result, dict):
        return result

    if name == "video":
        if result.get("manipulation_prob") is None and result.get("synthetic_prob") is not None:
            result["manipulation_prob"] = result["synthetic_prob"]
        if "signals" not in result:
            result["signals"] = _evidence_signals(result)

    elif name == "audio":
        transcript = ((result.get("transcription") or {}).get("text") or "").strip()
        result["transcript"] = transcript
        spoof = result.get("spoof_prob")
        signals = []
        if spoof is not None:
            level = "high" if spoof >= 0.6 else "low"
            signals.append(f"Synthetic-speech likelihood {spoof:.0%} ({level})")
        if transcript:
            signals.append("Speech transcribed successfully")
        result["signals"] = signals

    return result


# ---------------- Pipeline ----------------

def run_pipeline(case: dict) -> dict:
    """case = {"image": path|None, "video": path|None, "audio": path|None, "claim": str|None}"""
    results, notes = {}, []

    jobs = [("image", analyze_image, None)]
    for name, (module_path, func_name) in ANALYZERS.items():
        analyzer, load_error = _load(module_path, func_name)
        jobs.append((name, analyzer, load_error))

    for name, analyzer, load_error in jobs:
        path = case.get(name)
        if not path:
            continue
        if analyzer is None:
            notes.append(f"{name} analyzer not available: {load_error}")
            continue
        try:
            results[name] = _normalize(name, analyzer(path))
        except Exception as exc:
            notes.append(f"{name} analysis failed: {type(exc).__name__}: {exc}")

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