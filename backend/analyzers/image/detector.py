"""Pretrained AI-generated-image detector (CommunityForensics ViT).

The model is a stock ``ViTForImageClassification`` with a single logit. The
sigmoid of that logit is the model's probability that the image is
AI-generated; it is reported as-is (no rescaling) and is not calibrated.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Sequence
from typing import Any

from PIL import Image

from .errors import ImageAnalysisError, InferenceError, ModelLoadError
from .imaging import prepare_for_model
from .schema import LABEL_AI_GENERATED, LABEL_REAL

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
MODEL_ID_ENV = "TRUSTLAYER_IMAGE_MODEL"
MIN_TRANSFORMERS_VERSION = "5.4.0"
DEFAULT_THRESHOLD = 0.5
LOAD_RETRY_SECONDS = 30.0


class CommunityForensicsDetector:
    """Lazy-loading wrapper around the pretrained detector.

    The model is loaded on first use and then reused for every call on this
    instance. Use :func:`get_detector` to share one instance per process.
    """

    def __init__(
        self,
        model_id: str | None = None,
        *,
        device: str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        decision_threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.model_id = model_id or os.environ.get(MODEL_ID_ENV) or DEFAULT_MODEL_ID
        self._requested_device = device
        self._revision = revision
        self._cache_dir = cache_dir
        self._local_files_only = local_files_only
        self.decision_threshold = decision_threshold
        self._lock = threading.Lock()
        self._model: Any = None
        self._processor: Any = None
        self._device: str | None = None
        self._failure: tuple[float, ModelLoadError] | None = None

    @property
    def loaded(self) -> bool:
        """Whether the model is resident in memory."""
        return self._model is not None

    @property
    def input_size(self) -> int | None:
        """Model input side length in pixels, if known (loads the model)."""
        self.load()
        for attr in ("crop_size", "size"):
            source = getattr(self._processor, attr, None)
            if source is None:
                continue
            for key in ("height", "shortest_edge"):
                value = source.get(key) if isinstance(source, dict) else getattr(source, key, None)
                if isinstance(value, int):
                    return value
        return None

    def load(self) -> None:
        """Load the model and processor once (thread-safe); raises ``ModelLoadError``."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if self._failure and time.monotonic() - self._failure[0] < LOAD_RETRY_SECONDS:
                raise self._failure[1]
            try:
                self._load_locked()
            except ModelLoadError as exc:
                self._failure = (time.monotonic(), exc)
                raise
            self._failure = None

    def _load_locked(self) -> None:
        try:
            import torch
            import transformers
            from packaging.version import Version
            from transformers import ViTForImageClassification, ViTImageProcessor
        except ImportError as exc:
            raise ModelLoadError(f"Required ML dependencies are missing: {exc}") from exc

        if Version(transformers.__version__) < Version(MIN_TRANSFORMERS_VERSION):
            raise ModelLoadError(
                f"transformers>={MIN_TRANSFORMERS_VERSION} is required for correct preprocessing "
                f"of this model (found {transformers.__version__})."
            )

        device = self._requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs: dict[str, Any] = {
            "revision": self._revision,
            "cache_dir": self._cache_dir,
            "local_files_only": self._local_files_only,
        }
        try:
            logger.info("Loading image detector %s on %s", self.model_id, device)
            model = ViTForImageClassification.from_pretrained(self.model_id, **kwargs)
            processor = ViTImageProcessor.from_pretrained(self.model_id, **kwargs)
            if getattr(model.config, "num_labels", None) != 1:
                raise ModelLoadError(
                    f"Expected a single-logit head (num_labels=1), got {model.config.num_labels}."
                )
            model.to(device)
            model.eval()
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load model {self.model_id!r}: {type(exc).__name__}: {exc}"
            ) from exc

        self._model, self._processor, self._device = model, processor, device
        logger.info("Image detector ready on %s", device)

    def model_info(self) -> dict[str, Any]:
        """Describe the loaded model configuration (loads the model)."""
        self.load()
        import torch
        import transformers

        config = self._model.config
        return {
            "name": self.model_id,
            "revision": getattr(config, "_commit_hash", None) or self._revision,
            "architecture": type(self._model).__name__,
            "num_labels": config.num_labels,
            "input_size": self.input_size,
            "device": self._device,
            "dtype": str(next(self._model.parameters()).dtype).replace("torch.", ""),
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        }

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Predict for one decoded PIL image."""
        return self.predict_batch([image])[0]

    def predict_batch(self, images: Sequence[Image.Image]) -> list[dict[str, Any]]:
        """Predict for several decoded PIL images in one forward pass."""
        if not images:
            return []
        self.load()
        import torch

        try:
            prepared = [prepare_for_model(img) for img in images]
        except ImageAnalysisError as exc:
            raise InferenceError(exc.message) from exc

        try:
            inputs = self._processor(
                images=[rgb for rgb, _ in prepared], return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.inference_mode():
                logits = self._model(**inputs).logits.reshape(-1).float()
                ai_prob = torch.sigmoid(logits)
                real_prob = torch.sigmoid(-logits)
            logits_list = logits.cpu().tolist()
            ai_list = ai_prob.cpu().tolist()
            real_list = real_prob.cpu().tolist()
        except Exception as exc:
            raise InferenceError(f"{type(exc).__name__}: {exc}") from exc

        results = []
        for (_, preprocessing), logit, p_ai, p_real in zip(
            prepared, logits_list, ai_list, real_list
        ):
            results.append(
                {
                    "label": LABEL_AI_GENERATED if p_ai >= self.decision_threshold else LABEL_REAL,
                    "decision_threshold": self.decision_threshold,
                    "ai_generated_probability": p_ai,
                    "real_probability": p_real,
                    "raw_logit": logit,
                    "model": self.model_id,
                    "probability_calibrated": False,
                    "preprocessing": preprocessing,
                }
            )
        return results


_REGISTRY: dict[tuple[Any, ...], CommunityForensicsDetector] = {}
_REGISTRY_LOCK = threading.Lock()


def get_detector(
    model_id: str | None = None,
    *,
    device: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> CommunityForensicsDetector:
    """Return the process-wide detector for this configuration, creating it if needed."""
    resolved = model_id or os.environ.get(MODEL_ID_ENV) or DEFAULT_MODEL_ID
    key = (resolved, device, revision, cache_dir, local_files_only)
    with _REGISTRY_LOCK:
        detector = _REGISTRY.get(key)
        if detector is None:
            detector = CommunityForensicsDetector(
                resolved,
                device=device,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            _REGISTRY[key] = detector
        return detector


def reset_detector_cache() -> None:
    """Drop all cached detectors (frees memory and clears load-failure cooldowns)."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()
