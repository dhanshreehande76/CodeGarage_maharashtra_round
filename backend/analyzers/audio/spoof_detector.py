"""Real speech anti-spoofing detector for TrustLayer audio evidence."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import torch

MODEL_ID = "Vansh180/deepfake-audio-wav2vec2"
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 4
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS


class AudioSpoofDetectionError(RuntimeError):
    """Raised when the audio spoof detector cannot run."""


def _get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any, str]:
    """Load the anti-spoofing model once and reuse it."""
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    device = _get_device()
    try:
        processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
        model.to(device)
        model.eval()
        return processor, model, device
    except Exception as exc:
        raise AudioSpoofDetectionError(
            f"Failed to load audio spoof model '{MODEL_ID}': {exc}"
        ) from exc


def _label_is_spoof(label: str, index: int, id2label: dict[Any, str]) -> bool:
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    spoof_terms = ("spoof", "fake", "synthetic", "deepfake", "generated", "ai")
    real_terms = ("bonafide", "bona_fide", "real", "genuine", "human", "authentic")
    if any(term in normalized for term in spoof_terms):
        return True
    if any(term in normalized for term in real_terms):
        return False

    # The referenced model is documented as bonafide/spoof. Keep this fallback
    # explicit rather than silently inventing a probability from an unknown label.
    labels = [str(id2label.get(i, "")).lower() for i in range(2)]
    if labels and any("spoof" in value or "fake" in value for value in labels):
        return index == next(
            i for i, value in enumerate(labels) if "spoof" in value or "fake" in value
        )
    raise AudioSpoofDetectionError(
        f"Unable to identify the spoof class from model labels: {id2label}"
    )


def detect_audio_spoof(audio_path: str | Path) -> dict[str, Any]:
    """Estimate the probability that speech audio is spoofed/deepfake.

    The model is intended for speech anti-spoofing. For long recordings the
    file is split into non-overlapping four-second windows and the window
    probabilities are averaged. This is a model score, not a calibrated claim
    of certainty and not a guarantee of authenticity.
    """
    path = Path(audio_path)
    if not path.exists():
        raise AudioSpoofDetectionError(f"Audio file not found: {path}")
    if not path.is_file():
        raise AudioSpoofDetectionError(f"Audio path is not a file: {path}")

    try:
        audio, sample_rate = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise AudioSpoofDetectionError(
            f"Unable to decode audio file: {path}: {exc}"
        ) from exc
    if audio.size == 0:
        raise AudioSpoofDetectionError(f"Audio file is empty: {path}")

    processor, model, device = _load_model()
    id2label = getattr(model.config, "id2label", {}) or {}
    if not id2label:
        raise AudioSpoofDetectionError("Audio spoof model has no class labels.")

    chunks = []
    for start in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[start : start + CHUNK_SAMPLES]
        if chunk.size == 0:
            continue
        if chunk.size < CHUNK_SAMPLES:
            chunk = torch.nn.functional.pad(
                torch.from_numpy(chunk), (0, CHUNK_SAMPLES - chunk.size)
            ).numpy()

        try:
            inputs = processor(
                chunk,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=CHUNK_SAMPLES,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1)[0]
        except Exception as exc:
            raise AudioSpoofDetectionError(
                f"Audio spoof inference failed for chunk starting at {start / sample_rate:.2f}s: {exc}"
            ) from exc

        labels = [str(id2label.get(i, f"LABEL_{i}")) for i in range(len(probabilities))]
        spoof_probability = 0.0
        for index, label in enumerate(labels):
            if _label_is_spoof(label, index, id2label):
                spoof_probability += float(probabilities[index].item())

        chunks.append(
            {
                "start": round(start / sample_rate, 3),
                "end": round(min(start + CHUNK_SAMPLES, len(audio)) / sample_rate, 3),
                "spoof_probability": round(spoof_probability, 6),
            }
        )

    if not chunks:
        raise AudioSpoofDetectionError("No analyzable audio chunks were produced.")

    spoof_probability = sum(c["spoof_probability"] for c in chunks) / len(chunks)
    return {
        "model": MODEL_ID,
        "score_type": "spoof_probability",
        "spoof_probability": round(float(spoof_probability), 6),
        "chunks_analyzed": len(chunks),
        "chunk_seconds": CHUNK_SECONDS,
        "sample_rate": sample_rate,
        "segments": chunks,
        "device": device,
    }


__all__ = ["AudioSpoofDetectionError", "detect_audio_spoof"]
