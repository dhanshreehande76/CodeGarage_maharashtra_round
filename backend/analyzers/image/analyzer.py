"""Image authenticity analyzer: ML detector + forensics + metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from . import forensics
from .detector import CommunityForensicsDetector, get_detector
from .errors import ImageAnalysisError
from .hashing import perceptual_hash, sha256_file
from .imaging import open_image, to_rgb
from .metadata import MetadataAnalyzer
from .schema import (
    LABEL_AI_GENERATED,
    STATUS_ERROR,
    STATUS_PARTIAL,
    STATUS_SUCCESS,
    evidence_item,
    new_result,
)

logger = logging.getLogger(__name__)

_BOUNDARY_HIGH_UNCERTAINTY = 0.15
_BOUNDARY_MEDIUM_UNCERTAINTY = 0.35
_LOW_JPEG_QUALITY = 60
_ML_SEVERITY_BANDS = ((0.9, "high"), (0.7, "medium"), (0.5, "low"))
_LEVELS = ("low", "medium", "high")


@dataclass(frozen=True)
class ImageAnalyzerConfig:
    """Tunable limits and options for :class:`ImageAnalyzer`."""

    max_file_size_bytes: int = 100 * 1024 * 1024
    max_pixels: int = 64_000_000
    ela_quality: int = 90
    ela_max_pixels: int = 16_000_000
    stats_max_side: int = 1024
    frequency_crop_size: int = 512
    run_detector: bool = True
    model_id: str | None = None
    device: str | None = None
    local_files_only: bool = False


class ImageAnalyzer:
    """Standalone image analyzer; independent of any web framework.

    Example:
        >>> result = ImageAnalyzer().analyze("photo.jpg")
        >>> result["prediction"]["ai_generated_probability"]
    """

    def __init__(
        self,
        config: ImageAnalyzerConfig | None = None,
        *,
        detector: CommunityForensicsDetector | None = None,
    ) -> None:
        self.config = config or ImageAnalyzerConfig()
        self._detector = detector
        self._metadata = MetadataAnalyzer()

    @property
    def detector(self) -> CommunityForensicsDetector:
        """The shared detector (created lazily; the model loads on first prediction)."""
        if self._detector is None:
            self._detector = get_detector(
                self.config.model_id,
                device=self.config.device,
                local_files_only=self.config.local_files_only,
            )
        return self._detector

    def analyze(
        self,
        image_path: str | Path,
        *,
        declared_mime_type: str | None = None,
        reference_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Analyze one image file and return the structured result.

        Never raises for bad input; failures are reported in ``errors`` and
        ``status`` (``success`` / ``partial`` / ``error``).
        """
        path = Path(image_path)
        result = new_result(path.name)
        try:
            self._analyze(path, result, declared_mime_type, reference_time)
        except Exception as exc:
            logger.exception("Unexpected failure analysing %s", path.name)
            result["errors"].append(
                {"code": "unexpected_error", "message": f"{type(exc).__name__}: {exc}", "stage": "analysis"}
            )
        result["status"] = self._final_status(result)
        return result

    def _analyze(
        self,
        path: Path,
        result: dict[str, Any],
        declared_mime_type: str | None,
        reference_time: datetime | None,
    ) -> None:
        try:
            image = open_image(
                path,
                max_file_size_bytes=self.config.max_file_size_bytes,
                max_pixels=self.config.max_pixels,
            )
        except ImageAnalysisError as exc:
            result["errors"].append(exc.to_dict())
            self._hash_file_if_possible(path, result)
            return

        try:
            sha256 = self._safe(result, "hashing", lambda: sha256_file(path))
            result["hashes"]["sha256"] = sha256
            self._analyze_decoded(path, image, sha256, result, declared_mime_type, reference_time)
        finally:
            image.close()

    def _analyze_decoded(
        self,
        path: Path,
        image: Image.Image,
        sha256: str | None,
        result: dict[str, Any],
        declared_mime_type: str | None,
        reference_time: datetime | None,
    ) -> None:
        cfg = self.config
        rgb_info = self._safe(result, "preprocessing", lambda: to_rgb(image))
        rgb = rgb_info[0] if rgb_info else None
        if rgb_info and rgb_info[1]["alpha_flattened"]:
            result["warnings"].append("Transparency was flattened onto a white background for analysis.")

        result["metadata"] = self._safe(
            result,
            "metadata",
            lambda: self._metadata.analyze(
                path,
                image,
                sha256=sha256,
                declared_mime_type=declared_mime_type,
                reference_time=reference_time,
            ),
        )
        if rgb is not None:
            result["hashes"]["perceptual_hash"] = self._safe(
                result, "hashing", lambda: perceptual_hash(rgb)
            )

        compression = self._safe(result, "forensics", lambda: forensics.compression_info(image))
        forensic = result["forensics"]
        forensic["compression"] = compression
        if rgb is not None:
            forensic["ela"] = self._safe(
                result,
                "forensics",
                lambda: forensics.error_level_analysis(
                    rgb, image.format, cfg.ela_quality, cfg.ela_max_pixels
                ),
            )
            forensic["image_statistics"] = self._safe(
                result,
                "forensics",
                lambda: forensics.image_statistics(rgb, cfg.stats_max_side),
            )
            forensic["frequency_analysis"] = self._safe(
                result,
                "forensics",
                lambda: forensics.frequency_analysis(rgb, cfg.frequency_crop_size),
            )

        input_size: int | None = None
        if cfg.run_detector:
            prediction = self._safe(result, "detector", lambda: self._predict(image))
            if prediction is not None:
                prediction, model_info = prediction
                result["prediction"] = prediction
                result["model_info"] = model_info
                input_size = model_info.get("input_size")

        result["uncertainty"] = self._assess_uncertainty(
            result["prediction"], image.size, compression, input_size, cfg.run_detector
        )
        result["evidence"] = self._build_evidence(result)

    def _predict(self, image: Image.Image) -> tuple[dict[str, Any], dict[str, Any]]:
        detector = self.detector
        prediction = detector.predict(image)
        return prediction, detector.model_info()

    def _safe(self, result: dict[str, Any], stage: str, func: Any) -> Any:
        """Run a stage; record a structured error and return ``None`` on failure."""
        try:
            return func()
        except ImageAnalysisError as exc:
            result["errors"].append(exc.to_dict())
        except Exception as exc:
            logger.exception("Stage %s failed", stage)
            result["errors"].append(
                {"code": f"{stage}_failed", "message": f"{type(exc).__name__}: {exc}", "stage": stage}
            )
        return None

    def _hash_file_if_possible(self, path: Path, result: dict[str, Any]) -> None:
        try:
            if path.is_file():
                result["hashes"]["sha256"] = sha256_file(path)
        except OSError:
            pass

    @staticmethod
    def _final_status(result: dict[str, Any]) -> str:
        if result["metadata"] is None:
            return STATUS_ERROR
        return STATUS_PARTIAL if result["errors"] else STATUS_SUCCESS

    @staticmethod
    def _assess_uncertainty(
        prediction: dict[str, Any] | None,
        size: tuple[int, int],
        compression: dict[str, Any] | None,
        input_size: int | None,
        detector_enabled: bool,
    ) -> dict[str, Any]:
        """Rule-based uncertainty indicator (not a statistical confidence interval).

        Starts from the distance between the model probability and the decision
        boundary, then raises the level by one step for each documented
        evidence-quality limitation.
        """
        if prediction is None:
            reason = (
                "The detector was not run." if not detector_enabled
                else "The detector produced no prediction; see errors."
            )
            return {"status": "unavailable", "reason": reason}

        margin = abs(prediction["ai_generated_probability"] - prediction["decision_threshold"])
        if margin < _BOUNDARY_HIGH_UNCERTAINTY:
            level = 2
        elif margin < _BOUNDARY_MEDIUM_UNCERTAINTY:
            level = 1
        else:
            level = 0
        reasons = [f"Model probability is {margin:.2f} away from the decision threshold."]

        factors: list[str] = []
        if input_size and min(size) < input_size:
            factors.append(
                f"Image shorter side ({min(size)}px) is below the model input size "
                f"({input_size}px) and was upscaled."
            )
        jpeg = (compression or {}).get("jpeg") or {}
        quality = jpeg.get("estimated_quality")
        if quality is not None and quality < _LOW_JPEG_QUALITY:
            factors.append(f"Heavy JPEG compression (estimated quality {quality}).")
        level = min(level + len(factors), 2)
        reasons.extend(factors)
        reasons.append("Model probabilities are not calibrated.")

        return {
            "status": _LEVELS[level],
            "reason": " ".join(reasons),
            "margin_from_threshold": margin,
            "quality_factors": factors,
            "probability_calibrated": False,
        }

    @staticmethod
    def _build_evidence(result: dict[str, Any]) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        prediction = result["prediction"]
        if prediction is not None:
            p_ai = prediction["ai_generated_probability"]
            if prediction["label"] == LABEL_AI_GENERATED:
                severity = next(s for bound, s in _ML_SEVERITY_BANDS if p_ai >= bound)
                text = (
                    f"Pretrained detector assigns {p_ai:.1%} probability that the image is "
                    "AI-generated (uncalibrated model output)."
                )
            else:
                severity = "info"
                text = (
                    f"Pretrained detector assigns {p_ai:.1%} probability of AI generation, "
                    "i.e. it leans towards a real image (uncalibrated model output)."
                )
            evidence.append(evidence_item("ml_detector", severity, text, p_ai))

        ela = result["forensics"].get("ela")
        if ela and ela.get("signal") and ela["applicability"] == "applicable":
            band = ela.get("band")
            if band in ("moderate", "high"):
                evidence.append(
                    evidence_item(
                        "ela",
                        "low",
                        f"ELA shows {ela['signal']['interpretation']}. This is a weak, "
                        "non-conclusive forensic cue.",
                        ela["signal"]["value"],
                    )
                )

        freq = result["forensics"].get("frequency_analysis")
        if freq and freq.get("available"):
            if freq["spectral_peak_prominence_log10"] >= freq["peak_prominence_threshold"]:
                evidence.append(
                    evidence_item(
                        "frequency_analysis",
                        "low",
                        f"Spectrum shows {freq['signal']['interpretation']}.",
                        freq["signal"]["value"],
                    )
                )

        jpeg = ((result["forensics"].get("compression") or {}).get("jpeg")) or {}
        quality = jpeg.get("estimated_quality")
        if quality is not None and quality < _LOW_JPEG_QUALITY:
            evidence.append(
                evidence_item(
                    "jpeg_compression",
                    "info",
                    f"Estimated JPEG quality is low ({quality}); heavy compression can reduce "
                    "the reliability of detector and forensic signals.",
                    quality,
                )
            )

        metadata = result["metadata"] or {}
        for finding in metadata.get("findings", []):
            if finding["severity"] == "info" and finding["code"] not in (
                "exif_absent",
                "exif_present",
            ):
                continue
            evidence.append(
                evidence_item(
                    "metadata",
                    finding["severity"],
                    finding["description"],
                    finding["code"],
                )
            )
        return evidence
