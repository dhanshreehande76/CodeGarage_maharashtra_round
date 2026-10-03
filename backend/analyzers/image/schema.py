"""Output contract of the image analyzer.

Every call to ``ImageAnalyzer.analyze`` returns a JSON-serialisable dict with
exactly the top-level keys in ``RESULT_KEYS``, even on failure, so downstream
consumers (video analyzer, fusion layer) can rely on a stable shape.
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0"
ANALYZER_VERSION = "1.0.0"

STATUS_SUCCESS = "success"
STATUS_PARTIAL = "partial"
STATUS_ERROR = "error"

LABEL_AI_GENERATED = "ai_generated"
LABEL_REAL = "real"

SEVERITIES = ("info", "low", "medium", "high")
METADATA_STATUSES = ("present", "absent", "inconsistent", "suspicious", "unavailable")
UNCERTAINTY_STATUSES = ("low", "medium", "high", "unavailable")

RESULT_KEYS = (
    "modality",
    "schema_version",
    "analyzer_version",
    "status",
    "file_name",
    "prediction",
    "model_info",
    "uncertainty",
    "metadata",
    "hashes",
    "forensics",
    "evidence",
    "warnings",
    "errors",
)


def new_result(file_name: str) -> dict[str, Any]:
    """Create an empty result with the full, stable key set."""
    return {
        "modality": "image",
        "schema_version": SCHEMA_VERSION,
        "analyzer_version": ANALYZER_VERSION,
        "status": STATUS_ERROR,
        "file_name": file_name,
        "prediction": None,
        "model_info": None,
        "uncertainty": {
            "status": "unavailable",
            "reason": "No analysis has been performed.",
        },
        "metadata": None,
        "hashes": {"sha256": None, "perceptual_hash": None},
        "forensics": {
            "compression": None,
            "ela": None,
            "image_statistics": None,
            "frequency_analysis": None,
        },
        "evidence": [],
        "warnings": [],
        "errors": [],
    }


def evidence_item(
    type_: str, severity: str, description: str, value: Any = None
) -> dict[str, Any]:
    """Build one evidence entry."""
    return {
        "type": type_,
        "severity": severity,
        "description": description,
        "value": value,
    }
