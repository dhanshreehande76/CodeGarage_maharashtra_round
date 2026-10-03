"""Structured exceptions for the image analyzer."""

from __future__ import annotations

from typing import Any


class ImageAnalysisError(Exception):
    """Base error carrying a machine-readable code and the stage that failed."""

    def __init__(self, code: str, message: str, stage: str = "analysis") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage

    def to_dict(self) -> dict[str, Any]:
        """Return the error in the analyzer's output format."""
        return {"code": self.code, "message": self.message, "stage": self.stage}


class ModelLoadError(ImageAnalysisError):
    """Raised when the pretrained detector cannot be loaded."""

    def __init__(self, message: str) -> None:
        super().__init__("model_load_failed", message, "detector")


class InferenceError(ImageAnalysisError):
    """Raised when detector inference fails."""

    def __init__(self, message: str) -> None:
        super().__init__("inference_failed", message, "detector")
