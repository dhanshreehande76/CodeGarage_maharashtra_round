"""Cross-modal evidence consistency for TrustLayer."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Similarity thresholds are deliberately conservative.
CONSISTENT_THRESHOLD = 0.65
INCONSISTENT_THRESHOLD = 0.35


@lru_cache(maxsize=1)
def _load_clip():
    """Load the real CLIP model once."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL,
        pretrained=CLIP_PRETRAINED,
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()

    return model, preprocess, tokenizer


@lru_cache(maxsize=1)
def _load_text_model():
    """Load the real sentence embedding model once."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(TEXT_MODEL)


def _clip_image_text_score(image_path: str | Path, claim: str) -> float:
    """Calculate cosine similarity between an image and a text claim using CLIP."""
    from PIL import Image

    model, preprocess, tokenizer = _load_clip()

    image = Image.open(image_path).convert("RGB")

    image_features = model.encode_image(
        preprocess(image).unsqueeze(0)
    )

    text_features = model.encode_text(
        tokenizer([claim])
    )

    image_features = image_features.detach().cpu().numpy()
    text_features = text_features.detach().cpu().numpy()

    image_features /= np.linalg.norm(
        image_features,
        axis=-1,
        keepdims=True,
    )

    text_features /= np.linalg.norm(
        text_features,
        axis=-1,
        keepdims=True,
    )

    # Cosine similarity is [-1, 1]; map it to [0, 1].
    cosine = float(image_features @ text_features.T)

    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def _text_similarity(text_a: str, text_b: str) -> float:
    """Calculate semantic similarity between two pieces of text."""
    model = _load_text_model()

    embeddings = model.encode(
        [text_a, text_b],
        normalize_embeddings=True,
    )

    score = float(np.dot(embeddings[0], embeddings[1]))

    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def _finding(score: float, pair: str, detail: str) -> dict[str, Any]:
    if score >= CONSISTENT_THRESHOLD:
        status = "consistent"
    elif score <= INCONSISTENT_THRESHOLD:
        status = "inconsistent"
    else:
        status = "uncertain"

    return {
        "pair": pair,
        "score": round(score, 4),
        "status": status,
        "finding": detail,
    }


def run_crossmodal(
    case: dict,
    modality_results: dict,
    text_result: dict | None,
) -> list:
    """
    Compare independently analyzed modalities.

    Returns:
        [
            {
                "pair": "image-text",
                "score": 0-1,
                "status": "...",
                "finding": "..."
            }
        ]
    """
    findings: list[dict[str, Any]] = []

    claim = (case.get("claim") or "").strip()

    # ---------------------------------------------------------
    # IMAGE <-> CLAIM
    # ---------------------------------------------------------
    image_path = case.get("image")

    if image_path and claim:
        try:
            score = _clip_image_text_score(image_path, claim)

            findings.append(
                _finding(
                    score,
                    "image-text",
                    (
                        f"CLIP semantic similarity between the image and claim "
                        f"is {score:.2f}."
                    ),
                )
            )
        except Exception as exc:
            findings.append(
                {
                    "pair": "image-text",
                    "score": 0.5,
                    "status": "unavailable",
                    "finding": f"Image-text comparison failed: {exc}",
                }
            )

    # ---------------------------------------------------------
    # AUDIO TRANSCRIPT <-> CLAIM
    # ---------------------------------------------------------
    audio_result = modality_results.get("audio")

    if audio_result and claim:
        transcription = audio_result.get("transcription") or {}
        transcript = (transcription.get("text") or "").strip()

        if transcript:
            try:
                score = _text_similarity(transcript, claim)

                findings.append(
                    _finding(
                        score,
                        "audio-text",
                        (
                            f"Semantic similarity between the audio transcript "
                            f"and claim is {score:.2f}."
                        ),
                    )
                )
            except Exception as exc:
                findings.append(
                    {
                        "pair": "audio-text",
                        "score": 0.5,
                        "status": "unavailable",
                        "finding": f"Audio-text comparison failed: {exc}",
                    }
                )

    return findings


__all__ = ["run_crossmodal"]