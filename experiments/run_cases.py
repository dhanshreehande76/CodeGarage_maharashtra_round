import json
from pathlib import Path

from backend.pipeline import run_pipeline

ROOT = Path("datasets/custom_cases")
IMG = {".jpg", ".jpeg", ".png", ".webp"}
AUD = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
VID = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def find(folder, exts):
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in exts:
            return str(p)
    return None


rows, dump = [], {}
for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    expected = d.name.split("_")[-1]  # case_001_authentic -> authentic
    claim_file = d / "claim.txt"
    claim = claim_file.read_text(encoding="utf-8").strip() if claim_file.exists() else None

    case = {"image": find(d, IMG), "audio": find(d, AUD), "video": find(d, VID), "claim": claim}
    result = run_pipeline(case)
    predicted = result["verdict"].replace("coordinated_synthetic", "coordinated")
    dump[d.name] = result
    rows.append((d.name, expected, predicted, result["trust_score"], result["confidence"],
                 [k for k, v in case.items() if v], result["notes"]))

print(f"\n{'case':32} {'expected':13} {'predicted':13} trust  conf  inputs")
for name, exp, pred, trust, conf, inputs, notes in rows:
    flag = "OK " if exp == pred else "XX "
    print(f"{flag}{name:29} {exp:13} {pred:13} {trust:<6} {conf:<5} {','.join(inputs)}")
    if notes:
        print("      notes:", notes)

correct = sum(1 for r in rows if r[1] == r[2])
print(f"\n{correct}/{len(rows)} match the expected label")

Path("tmp").mkdir(exist_ok=True)
Path("tmp/case_results.json").write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")
print("Full results saved to tmp/case_results.json")