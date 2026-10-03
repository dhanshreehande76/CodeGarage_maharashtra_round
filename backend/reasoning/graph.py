def build_graph(risks: dict, findings: list) -> dict:
    """Nodes = evidence sources, edges = cross-modal conflicts."""
    nodes = {name: {"id": name, "risk": round(r, 2)} for name, r in risks.items()}
    edges = []
    for f in findings:
        a, b = f["pair"].split("-", 1)
        conflict = round(1.0 - f["score"], 2)
        edges.append({"from": a, "to": b, "conflict": conflict})
        for n in (a, b):
            nodes.setdefault(n, {"id": n, "risk": 0.0})

    # Sources without their own detector (text, metadata) take the strongest conflict touching them
    for name, node in nodes.items():
        if name not in risks:
            touching = [e["conflict"] for e in edges if name in (e["from"], e["to"])]
            node["risk"] = max(touching, default=0.0)

    return {"nodes": list(nodes.values()), "edges": edges}