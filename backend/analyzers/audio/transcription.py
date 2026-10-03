"""Whisper-based speech transcription for the TrustLayer audio analyzer."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import torch

MODEL_ID = "openai/whisper-small"
SAMPLE_RATE = 16_000


class AudioTranscriptionError(RuntimeError):
    """Raised when audio transcription cannot be completed."""


def _get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _get_pipeline():
    """Load Whisper once and reuse it for subsequent files."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = _get_device()
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID)
        model.to(device)
        model.eval()

        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            chunk_length_s=30,
        )
    except Exception as exc:
        raise AudioTranscriptionError(
            f"Failed to load Whisper model '{MODEL_ID}': {exc}"
        ) from exc


def _normalise_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def transcribe_audio(audio_path: str | Path) -> dict[str, Any]:
    """Transcribe an audio file with the real Whisper model."""
    path = Path(audio_path)
    if not path.exists():
        raise AudioTranscriptionError(f"Audio file not found: {path}")
    if not path.is_file():
        raise AudioTranscriptionError(f"Audio path is not a file: {path}")

    try:
        audio, sample_rate = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise AudioTranscriptionError(
            f"Unable to decode audio file: {path}: {exc}"
        ) from exc

    if audio.size == 0:
        raise AudioTranscriptionError(f"Audio file is empty: {path}")

    duration = float(len(audio) / sample_rate)

    try:
        result = _get_pipeline()(
            {"array": audio, "sampling_rate": sample_rate},
            return_timestamps=True,
        )
    except Exception as exc:
        raise AudioTranscriptionError(
            f"Whisper transcription failed for {path}: {exc}"
        ) from exc

    segments: list[dict[str, Any]] = []
    for chunk in result.get("chunks", []):
        timestamp = chunk.get("timestamp")
        start = end = None
        if timestamp:
            start, end = timestamp
        segments.append(
            {
                "text": chunk.get("text", "").strip(),
                "start": _normalise_timestamp(start),
                "end": _normalise_timestamp(end),
            }
        )

    return {
        "text": result.get("text", "").strip(),
        "language": result.get("language"),
        "duration_seconds": round(duration, 3),
        "sample_rate": sample_rate,
        "segments": segments,
        "model": MODEL_ID,
    }
