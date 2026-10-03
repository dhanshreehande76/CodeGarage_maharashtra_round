from __future__ import annotations

import numpy as np
import soundfile as sf
import torch

from backend.analyzers.audio.spoof_detector import detect_audio_spoof


def test_spoof_detector_averages_chunk_scores(monkeypatch, tmp_path):
    path = tmp_path / "speech.wav"
    sf.write(path, np.zeros(5 * 16_000, dtype=np.float32), 16_000)

    class FakeProcessor:
        def __call__(self, chunk, **kwargs):
            assert kwargs["sampling_rate"] == 16_000
            return {"input_values": torch.zeros(1, 64_000)}

    class FakeConfig:
        id2label = {0: "bonafide", 1: "spoof"}

    class FakeModel:
        config = FakeConfig()
        calls = 0

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            self.calls += 1
            # Equal logits => 0.5 spoof probability per chunk.
            return type("Output", (), {"logits": torch.tensor([[0.0, 0.0]])})()

    model = FakeModel()
    monkeypatch.setattr(
        "backend.analyzers.audio.spoof_detector._load_model",
        lambda: (FakeProcessor(), model, "cpu"),
    )

    result = detect_audio_spoof(path)
    assert result["model"] == "Vansh180/deepfake-audio-wav2vec2"
    assert result["chunks_analyzed"] == 2
    assert result["spoof_probability"] == 0.5
    assert model.calls == 2
