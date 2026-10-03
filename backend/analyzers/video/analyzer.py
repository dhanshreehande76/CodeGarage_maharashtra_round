"""Video deepfake analyzer for TrustLayer."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from decord import VideoReader, cpu
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor


MODEL_ID = "Vansh180/VideoMae-ffc23-deepfake-detector"
NUM_FRAMES = 16
IMAGE_SIZE = 224


class VideoAnalysisError(RuntimeError):
    """Raised when video analysis cannot be completed."""


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any, str]:
    """Load the pretrained VideoMAE detector once."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    try:
        processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
        model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)
        model.to(device)
        model.eval()
        return processor, model, device
    except Exception as exc:
        raise VideoAnalysisError(
            f"Failed to load video model '{MODEL_ID}': {exc}"
        ) from exc


def _get_video_frames(path: Path) -> tuple[list[Any], float, int]:
    """Read evenly spaced frames from a video."""
    try:
        reader = VideoReader(str(path), ctx=cpu(0))
    except Exception as exc:
        raise VideoAnalysisError(
            f"Unable to open video '{path}': {exc}"
        ) from exc

    frame_count = len(reader)

    if frame_count == 0:
        raise VideoAnalysisError("Video contains no frames.")

    fps = float(reader.get_avg_fps())

    indices = torch.linspace(
        0,
        frame_count - 1,
        steps=min(NUM_FRAMES, frame_count),
    ).long().tolist()

    try:
        frames = reader.get_batch(indices).asnumpy()
    except Exception as exc:
        raise VideoAnalysisError(
            f"Unable to extract video frames: {exc}"
        ) from exc

    return list(frames), fps, frame_count


def _is_fake_label(label: str) -> bool:
    normalized = label.lower().replace("-", "_").replace(" ", "_")

    fake_terms = (
        "fake",
        "deepfake",
        "synthetic",
        "manipulated",
        "generated",
    )

    return any(term in normalized for term in fake_terms)


def analyze_video(video_path: str | Path) -> dict[str, Any]:
    """Analyze a video with a pretrained VideoMAE deepfake detector."""
    path = Path(video_path)

    if not path.exists():
        raise VideoAnalysisError(f"Video file not found: {path}")

    if not path.is_file():
        raise VideoAnalysisError(f"Video path is not a file: {path}")

    frames, fps, frame_count = _get_video_frames(path)

    processor, model, device = _load_model()

    try:
        inputs = processor(
            frames,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]

    except Exception as exc:
        raise VideoAnalysisError(
            f"Video model inference failed: {exc}"
        ) from exc

    id2label = getattr(model.config, "id2label", {}) or {}

    labels = [
        str(id2label.get(index, f"LABEL_{index}"))
        for index in range(len(probabilities))
    ]

    fake_probability = 0.0

    for index, label in enumerate(labels):
        if _is_fake_label(label):
            fake_probability += float(probabilities[index].item())

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_label = labels[predicted_index]

    if fake_probability >= 0.5:
        verdict = "manipulated"
    else:
        verdict = "authentic"

    return {
        "modality": "video",
        "status": "success",
        "file_name": path.name,
        "model": MODEL_ID,
        "synthetic_prob": round(fake_probability, 6),
        "predicted_label": predicted_label,
        "verdict": verdict,
        "frame_count": frame_count,
        "frames_analyzed": len(frames),
        "fps": round(fps, 3),
        "device": device,
        "evidence": [
            {
                "type": "video_deepfake_detection",
                "severity": (
                    "high"
                    if fake_probability >= 0.8
                    else "medium"
                    if fake_probability >= 0.6
                    else "low"
                ),
                "description": "VideoMAE deepfake detector score.",
                "value": {
                    "model": MODEL_ID,
                    "synthetic_probability": round(
                        fake_probability,
                        6,
                    ),
                    "predicted_label": predicted_label,
                },
            }
        ],
        "warnings": [
            "Model output is an automated forensic signal, not definitive proof of authenticity."
        ],
        "errors": [],
    }


__all__ = [
    "VideoAnalysisError",
    "analyze_video",
]