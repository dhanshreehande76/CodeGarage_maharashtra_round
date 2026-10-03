"""Audio analyzer entry point used by the TrustLayer pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .spoof_detector import AudioSpoofDetectionError, detect_audio_spoof
from .transcription import AudioTranscriptionError, transcribe_audio

logger = logging.getLogger(__name__)


def analyze_audio(audio_path: str | Path) -> dict[str, Any]:
    """Run Whisper transcription and real speech anti-spoofing analysis."""
    path = Path(audio_path)
    errors: list[dict[str, str]] = []

    try:
        transcription = transcribe_audio(path)
    except AudioTranscriptionError as exc:
        raise AudioTranscriptionError(str(exc)) from exc

    try:
        spoof = detect_audio_spoof(path)
    except AudioSpoofDetectionError as exc:
        logger.warning("Audio spoof detection failed for %s: %s", path, exc)
        errors.append(
            {
                "code": "spoof_detection_failed",
                "message": str(exc),
                "stage": "spoof_detector",
            }
        )
        spoof = None

    evidence: list[dict[str, Any]] = [
        {
            "type": "transcription",
            "severity": "info",
            "description": "Speech was transcribed with Whisper.",
            "value": {
                "model": transcription["model"],
                "text": transcription["text"],
            },
        }
    ]

    spoof_prob = None
    if spoof is not None:
        spoof_prob = spoof["spoof_probability"]
        severity = "high" if spoof_prob >= 0.8 else "medium" if spoof_prob >= 0.6 else "low"
        evidence.append(
            {
                "type": "audio_spoof_detection",
                "severity": severity,
                "description": "Speech anti-spoofing model score.",
                "value": spoof,
            }
        )

    return {
        "modality": "audio",
        "status": "success" if not errors else "partial",
        "file_name": path.name,
        "transcription": transcription,
        "spoof_prob": spoof_prob,
        "spoof_detection": spoof,
        "evidence": evidence,
        "warnings": [],
        "errors": errors,
    }


__all__ = [
    "AudioSpoofDetectionError",
    "AudioTranscriptionError",
    "analyze_audio",
    "detect_audio_spoof",
    "transcribe_audio",
]
