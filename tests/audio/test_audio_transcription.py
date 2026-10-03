from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from backend.analyzers.audio.analyzer import analyze_audio
from backend.analyzers.audio.transcription import (
    AudioTranscriptionError,
    transcribe_audio,
)


def test_missing_audio_file():
    with pytest.raises(AudioTranscriptionError, match="Audio file not found"):
        transcribe_audio("does-not-exist.wav")


def test_audio_path_must_be_file(tmp_path):
    with pytest.raises(AudioTranscriptionError, match="not a file"):
        transcribe_audio(tmp_path)


def test_empty_audio_file_is_rejected(tmp_path):
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    with pytest.raises(AudioTranscriptionError, match="Unable to decode audio file"):
        transcribe_audio(path)


def test_transcription_result_shape(monkeypatch, tmp_path):
    path = tmp_path / "speech.wav"
    samples = np.zeros(16_000, dtype=np.float32)
    sf.write(path, samples, 16_000)

    class FakePipeline:
        def __call__(self, audio_input, return_timestamps=False):
            assert return_timestamps is True
            assert audio_input["sampling_rate"] == 16_000
            return {
                "text": " hello world ",
                "language": "en",
                "chunks": [
                    {"text": " hello ", "timestamp": (0.0, 0.5)},
                    {"text": " world ", "timestamp": (0.5, 1.0)},
                ],
            }

    monkeypatch.setattr(
        "backend.analyzers.audio.transcription._get_pipeline",
        lambda: FakePipeline(),
    )

    result = transcribe_audio(path)
    assert result["text"] == "hello world"
    assert result["language"] == "en"
    assert result["duration_seconds"] == 1.0
    assert result["sample_rate"] == 16_000
    assert result["segments"] == [
        {"text": "hello", "start": 0.0, "end": 0.5},
        {"text": "world", "start": 0.5, "end": 1.0},
    ]
    assert result["model"] == "openai/whisper-small"
    json.dumps(result, allow_nan=False)


def test_analyze_audio_does_not_treat_transcription_as_spoof_evidence(monkeypatch, tmp_path):
    path = tmp_path / "speech.wav"
    path.write_bytes(b"placeholder")

    transcription = {
        "text": "hello world",
        "language": "en",
        "duration_seconds": 1.0,
        "sample_rate": 16_000,
        "segments": [],
        "model": "openai/whisper-small",
    }
    monkeypatch.setattr(
        "backend.analyzers.audio.analyzer.transcribe_audio",
        lambda _: transcription,
    )
    monkeypatch.setattr(
        "backend.analyzers.audio.analyzer.detect_audio_spoof",
        lambda _: {"model": "test", "spoof_probability": 0.2},
    )

    result = analyze_audio(path)
    assert result["modality"] == "audio"
    assert result["status"] == "success"
    assert result["spoof_prob"] == 0.2
    assert result["transcription"] == transcription
    assert result["evidence"][0]["type"] == "transcription"
