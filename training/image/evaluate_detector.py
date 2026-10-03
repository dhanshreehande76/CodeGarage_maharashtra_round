"""Evaluate the pretrained image detector on a labelled real/fake dataset.

Dataset layout (folders are searched recursively; ``fake`` is the positive class)::

    <data-dir>/
        real/   authentic images
        fake/   AI-generated / manipulated images

Usage::

    python training/image/evaluate_detector.py --data-dir datasets/raw/image_eval

Only metrics computed from actual model outputs are written. If the dataset is
missing or has an empty class, the script exits without producing any metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.analyzers.image.detector import CommunityForensicsDetector  # noqa: E402
from backend.analyzers.image.errors import ImageAnalysisError  # noqa: E402
from backend.analyzers.image.imaging import SUPPORTED_EXTENSIONS, open_image  # noqa: E402

logger = logging.getLogger("evaluate_detector")

CLASS_DIRS = {"real": 0, "fake": 1}
EXIT_BAD_DATASET = 2
EXIT_RUNTIME_FAILURE = 3


def collect_samples(data_dir: Path, limit: int | None) -> list[tuple[Path, int]]:
    """List ``(path, label)`` pairs in deterministic order; label 1 = fake."""
    samples: list[tuple[Path, int]] = []
    for name, label in CLASS_DIRS.items():
        folder = data_dir / name
        if not folder.is_dir():
            continue
        files = sorted(
            p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if limit is not None:
            files = files[:limit]
        samples.extend((p, label) for p in files)
    return samples


def batched(items: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run_inference(
    detector: CommunityForensicsDetector,
    samples: list[tuple[Path, int]],
    batch_size: int,
    max_pixels: int,
    max_file_size_bytes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for batch in batched(samples, batch_size):
        images, kept = [], []
        for path, label in batch:
            try:
                images.append(
                    open_image(path, max_file_size_bytes=max_file_size_bytes, max_pixels=max_pixels)
                )
                kept.append((path, label))
            except ImageAnalysisError as exc:
                skipped.append({"path": str(path), "code": exc.code, "message": exc.message})
        try:
            predictions = detector.predict_batch(images) if images else []
        finally:
            for image in images:
                image.close()
        for (path, label), pred in zip(kept, predictions):
            rows.append(
                {
                    "path": str(path),
                    "label": label,
                    "ai_generated_probability": pred["ai_generated_probability"],
                    "raw_logit": pred["raw_logit"],
                }
            )
        logger.info("Processed %d/%d", len(rows) + len(skipped), len(samples))
    return rows, skipped


def compute_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Compute classification metrics (positive class = fake)."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_true = [r["label"] for r in rows]
    scores = [r["ai_generated_probability"] for r in rows]
    y_pred = [int(s >= threshold) for s in scores]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    return {
        "threshold": threshold,
        "positive_class": "fake",
        "n_samples": len(rows),
        "n_real": y_true.count(0),
        "n_fake": y_true.count(1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "confusion_matrix": {
            "labels": ["real", "fake"],
            "matrix": matrix.tolist(),
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing real/ and fake/")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "experiments" / "image_detector_eval")
    parser.add_argument("--model", default=None, help="Hugging Face model id or local path (default: CommunityForensics ViT)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold on ai_generated_probability")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Max images per class (for quick runs)")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--local-files-only", action="store_true", help="Never contact the Hugging Face Hub")
    parser.add_argument("--max-pixels", type=int, default=64_000_000)
    parser.add_argument("--max-file-size-bytes", type=int, default=100 * 1024 * 1024)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.data_dir.is_dir():
        logger.error("Dataset folder not found: %s (see training/image/README.md)", args.data_dir)
        return EXIT_BAD_DATASET
    samples = collect_samples(args.data_dir, args.limit)
    labels = [label for _, label in samples]
    if 0 not in labels or 1 not in labels:
        logger.error(
            "Need at least one image in both real/ and fake/ under %s (found real=%d, fake=%d).",
            args.data_dir, labels.count(0), labels.count(1),
        )
        return EXIT_BAD_DATASET

    detector = CommunityForensicsDetector(
        args.model,
        device=args.device,
        local_files_only=args.local_files_only,
        decision_threshold=args.threshold,
    )
    try:
        model_info = detector.model_info()
        rows, skipped = run_inference(
            detector, samples, args.batch_size, args.max_pixels, args.max_file_size_bytes
        )
    except ImageAnalysisError as exc:
        logger.error("%s: %s", exc.code, exc.message)
        return EXIT_RUNTIME_FAILURE

    present = {r["label"] for r in rows}
    if present != {0, 1}:
        logger.error("After skipping unreadable files one class is empty; cannot compute metrics.")
        return EXIT_BAD_DATASET

    metrics = compute_metrics(rows, args.threshold)
    report = {
        "model": model_info,
        "dataset": {"data_dir": str(args.data_dir), "skipped_files": skipped, "n_skipped": len(skipped)},
        "metrics": metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "ai_generated_probability", "raw_logit"])
        writer.writeheader()
        writer.writerows(rows)

    cm = metrics["confusion_matrix"]
    print(f"samples={metrics['n_samples']} (real={metrics['n_real']}, fake={metrics['n_fake']}), skipped={len(skipped)}")
    print(f"accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
          f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f}")
    print(f"confusion matrix [rows=true real/fake, cols=pred real/fake]: {cm['matrix']}")
    print(f"written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
