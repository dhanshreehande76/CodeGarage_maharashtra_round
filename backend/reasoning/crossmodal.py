def run_crossmodal(case: dict, modality_results: dict, text_result: dict | None) -> list:
    """
    Returns findings: [{"pair": "image-text", "score": 0-1 (1 = consistent), "finding": str}]
    TODO (Step 4): CLIP image<->text, transcript<->claim, metadata<->claim.
    """
    return []