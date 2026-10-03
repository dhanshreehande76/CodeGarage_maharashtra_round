"""EXIF and file-level metadata analysis.

Findings describe what is present, absent or inconsistent. Missing EXIF is
common (screenshots, social-media re-encoding, privacy stripping, PNG/WebP
exports) and is never treated as evidence of AI generation.
"""

from __future__ import annotations

import logging
import mimetypes
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

from .imaging import FORMAT_EXTENSIONS

logger = logging.getLogger(__name__)

_EXIF_IFD = 0x8769
_GPS_IFD = 0x8825
_TAG_MAKE = 0x010F
_TAG_MODEL = 0x0110
_TAG_SOFTWARE = 0x0131
_TAG_DATETIME = 0x0132
_TAG_ORIENTATION = 0x0112
_TAG_DT_ORIGINAL = 0x9003
_TAG_DT_DIGITIZED = 0x9004
_TAG_PIXEL_X = 0xA002
_TAG_PIXEL_Y = 0xA003
_TAG_MAKER_NOTE = 0x927C

_EXIF_FORMATS = frozenset({"JPEG", "MPO", "PNG", "WEBP", "TIFF"})
_EXIF_DATETIME = re.compile(r"^(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})$")
_MAX_STRING = 256
_MAX_BYTES_INLINE = 64
_FUTURE_TOLERANCE = timedelta(days=1)
_C2PA_MARKER = b"c2pa"

_GENERATOR_PATTERN = re.compile(
    r"stable[ _-]?diffusion|midjourney|dall[-_ ·]?e|firefly|comfyui|automatic1111|"
    r"invokeai|novelai|\bsdxl\b|leonardo\.ai|ideogram|nano[ _-]?banana",
    re.IGNORECASE,
)
_PNG_TEXT_KEYS = ("parameters", "prompt", "workflow", "Software", "Comment", "Description")
_A1111_SIGNATURE = re.compile(r"Steps:\s*\d+.*Sampler:", re.DOTALL)


def _finding(code: str, category: str, severity: str, description: str, value: Any = None) -> dict[str, Any]:
    return {
        "code": code,
        "category": category,
        "severity": severity,
        "description": description,
        "value": value,
    }


def _sanitize(value: Any) -> Any:
    """Convert an EXIF value to a JSON-serialisable form."""
    if isinstance(value, bytes):
        if len(value) > _MAX_BYTES_INLINE:
            return f"<{len(value)} bytes>"
        text = value.decode("utf-8", errors="replace").rstrip("\x00 ")
        return text if text.isprintable() else f"<{len(value)} bytes>"
    if isinstance(value, str):
        return value.rstrip("\x00 ")[:_MAX_STRING]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, (tuple, list)):
        return [_sanitize(v) for v in value[:32]]
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)[:_MAX_STRING]


def _text(value: Any) -> str | None:
    cleaned = _sanitize(value)
    if isinstance(cleaned, str) and cleaned.strip():
        return cleaned.strip()
    return None


def _parse_exif_datetime(raw: Any) -> tuple[datetime | None, str | None]:
    """Parse an EXIF timestamp; return (value, problem)."""
    text = _text(raw)
    if text is None:
        return None, "empty"
    match = _EXIF_DATETIME.match(text)
    if not match:
        return None, "malformed"
    parts = tuple(int(p) for p in match.groups())
    if parts[:3] == (0, 0, 0):
        return None, "placeholder"
    try:
        return datetime(*parts), None
    except ValueError:
        return None, "invalid"


def _dms_to_decimal(values: Any, ref: Any) -> float | None:
    try:
        degrees, minutes, seconds = (float(v) for v in values)
    except (TypeError, ValueError):
        return None
    ref_text = _text(ref)
    if ref_text not in ("N", "S", "E", "W"):
        return None
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    return -decimal if ref_text in ("S", "W") else decimal


def _named_tags(ifd: Any) -> dict[str, Any]:
    named: dict[str, Any] = {}
    for tag, value in dict(ifd).items():
        if tag == _TAG_MAKER_NOTE:
            continue
        named[TAGS.get(tag, f"Tag_{tag:#06x}")] = _sanitize(value)
    return named


def _contains_marker(path: Path, marker: bytes) -> bool:
    overlap = len(marker) - 1
    tail = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if marker in tail + chunk:
                return True
            tail = chunk[-overlap:] if overlap else b""
    return False


class MetadataAnalyzer:
    """Extract EXIF/file metadata and report structural findings."""

    def analyze(
        self,
        path: Path,
        image: Image.Image,
        *,
        sha256: str | None = None,
        declared_mime_type: str | None = None,
        reference_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Analyze metadata of an already opened image.

        Args:
            path: File on disk.
            image: Decoded image opened from ``path``.
            sha256: Precomputed file hash to embed in the result.
            declared_mime_type: Content-Type claimed by an uploader, if any.
            reference_time: "Now" used for future-timestamp checks (injectable for tests).
        """
        now = reference_time or datetime.now()
        findings: list[dict[str, Any]] = []
        fmt = image.format or ""
        extension = path.suffix.lower()
        detected_mime = Image.MIME.get(fmt)
        extension_mime = mimetypes.guess_type(path.name)[0]

        width, height = image.size
        exif_block: dict[str, Any] = {
            "available": False,
            "camera_make": None,
            "camera_model": None,
            "software": None,
            "datetime": None,
            "datetime_original": None,
            "datetime_digitized": None,
            "orientation": None,
            "gps": None,
            "tags": {},
        }

        self._check_format_consistency(
            findings, fmt, extension, detected_mime, extension_mime, declared_mime_type
        )
        self._check_dimensions(findings, width, height)

        exif_status = self._read_exif(image, fmt, exif_block, findings, now)
        text_chunks = self._read_text_chunks(image, findings)
        if _contains_marker(path, _C2PA_MARKER):
            findings.append(
                _finding(
                    "content_credentials_marker",
                    "present",
                    "info",
                    "File contains a C2PA/Content Credentials marker. Presence only; the "
                    "manifest signature was not verified.",
                )
            )

        status = self._overall_status(exif_status, findings)
        return {
            "status": status,
            "file_name": path.name,
            "file_extension": extension,
            "format": fmt or None,
            "mime_type": detected_mime,
            "extension_mime_type": extension_mime,
            "declared_mime_type": declared_mime_type,
            "color_mode": image.mode,
            "width": width,
            "height": height,
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256,
            "exif": exif_block,
            "text_chunks": text_chunks,
            "findings": findings,
        }

    def _check_format_consistency(
        self,
        findings: list[dict[str, Any]],
        fmt: str,
        extension: str,
        detected_mime: str | None,
        extension_mime: str | None,
        declared_mime: str | None,
    ) -> None:
        allowed = FORMAT_EXTENSIONS.get(fmt, frozenset())
        if not extension:
            findings.append(
                _finding("extension_missing", "absent", "info", "File has no extension.")
            )
        elif allowed and extension not in allowed:
            findings.append(
                _finding(
                    "extension_format_mismatch",
                    "inconsistent",
                    "low",
                    f"Extension {extension!r} does not match the detected format {fmt}.",
                    {"extension": extension, "detected_format": fmt},
                )
            )

        if extension_mime and detected_mime and extension_mime != detected_mime:
            if not (fmt == "MPO" and extension_mime == "image/jpeg"):
                findings.append(
                    _finding(
                        "mime_extension_mismatch",
                        "inconsistent",
                        "low",
                        f"MIME type implied by the extension ({extension_mime}) differs from "
                        f"the detected content type ({detected_mime}).",
                        {"extension_mime": extension_mime, "detected_mime": detected_mime},
                    )
                )
        if declared_mime and detected_mime and declared_mime.lower() != detected_mime:
            if not (fmt == "MPO" and declared_mime.lower() == "image/jpeg"):
                findings.append(
                    _finding(
                        "mime_declared_mismatch",
                        "inconsistent",
                        "low",
                        f"Declared MIME type ({declared_mime}) differs from the detected "
                        f"content type ({detected_mime}).",
                        {"declared_mime": declared_mime, "detected_mime": detected_mime},
                    )
                )

    def _check_dimensions(self, findings: list[dict[str, Any]], width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            findings.append(
                _finding(
                    "invalid_dimensions",
                    "inconsistent",
                    "medium",
                    f"Image reports impossible dimensions {width}x{height}.",
                )
            )

    def _read_exif(
        self,
        image: Image.Image,
        fmt: str,
        block: dict[str, Any],
        findings: list[dict[str, Any]],
        now: datetime,
    ) -> str:
        if fmt not in _EXIF_FORMATS:
            findings.append(
                _finding(
                    "exif_unavailable",
                    "unavailable",
                    "info",
                    f"The {fmt or 'unknown'} format does not carry EXIF metadata.",
                )
            )
            return "unavailable"

        try:
            exif = image.getexif()
            base = dict(exif)
        except Exception as exc:
            logger.warning("EXIF parsing failed: %s", exc)
            findings.append(
                _finding(
                    "exif_malformed",
                    "inconsistent",
                    "low",
                    f"EXIF block could not be parsed: {type(exc).__name__}.",
                )
            )
            return "malformed"

        exif_ifd: dict[int, Any] = {}
        gps_ifd: dict[int, Any] = {}
        try:
            exif_ifd = dict(exif.get_ifd(_EXIF_IFD))
        except Exception as exc:
            findings.append(
                _finding("exif_ifd_malformed", "inconsistent", "low", f"Exif sub-IFD unreadable: {type(exc).__name__}.")
            )
        try:
            gps_ifd = dict(exif.get_ifd(_GPS_IFD))
        except Exception as exc:
            findings.append(
                _finding("gps_ifd_malformed", "inconsistent", "low", f"GPS IFD unreadable: {type(exc).__name__}.")
            )

        if not base and not exif_ifd and not gps_ifd:
            findings.append(
                _finding(
                    "exif_absent",
                    "absent",
                    "info",
                    "No EXIF metadata found. This is common for screenshots, edited, re-encoded "
                    "or privacy-stripped images and is not evidence of AI generation.",
                )
            )
            return "absent"

        block["available"] = True
        merged = {**base, **exif_ifd}
        block["tags"] = _named_tags(merged)
        block["camera_make"] = _text(merged.get(_TAG_MAKE))
        block["camera_model"] = _text(merged.get(_TAG_MODEL))
        block["software"] = _text(merged.get(_TAG_SOFTWARE))
        block["datetime"] = _text(merged.get(_TAG_DATETIME))
        block["datetime_original"] = _text(merged.get(_TAG_DT_ORIGINAL))
        block["datetime_digitized"] = _text(merged.get(_TAG_DT_DIGITIZED))
        orientation = merged.get(_TAG_ORIENTATION)
        block["orientation"] = orientation if isinstance(orientation, int) else None

        findings.append(
            _finding("exif_present", "present", "info", "EXIF metadata is present.", len(merged))
        )
        if block["camera_make"] or block["camera_model"]:
            findings.append(
                _finding(
                    "camera_info_present",
                    "present",
                    "info",
                    "Camera make/model recorded.",
                    {"make": block["camera_make"], "model": block["camera_model"]},
                )
            )
        else:
            findings.append(
                _finding(
                    "camera_info_absent",
                    "absent",
                    "info",
                    "EXIF is present but has no camera make or model.",
                )
            )

        self._check_software(findings, block["software"])
        self._check_timestamps(findings, merged, now)
        self._check_orientation(findings, orientation)
        self._check_gps(findings, block, gps_ifd)
        self._check_exif_dimensions(findings, image, merged, orientation)
        return "present"

    def _check_software(self, findings: list[dict[str, Any]], software: str | None) -> None:
        if not software:
            return
        findings.append(
            _finding("software_tag_present", "present", "info", f"Software tag recorded: {software!r}.", software)
        )
        if _GENERATOR_PATTERN.search(software):
            findings.append(
                _finding(
                    "generator_software_tag",
                    "suspicious",
                    "medium",
                    f"Software tag names an image-generation tool: {software!r}.",
                    software,
                )
            )

    def _check_timestamps(
        self, findings: list[dict[str, Any]], tags: dict[int, Any], now: datetime
    ) -> None:
        parsed: dict[str, datetime] = {}
        for label, tag in (
            ("DateTime", _TAG_DATETIME),
            ("DateTimeOriginal", _TAG_DT_ORIGINAL),
            ("DateTimeDigitized", _TAG_DT_DIGITIZED),
        ):
            if tag not in tags:
                continue
            value, problem = _parse_exif_datetime(tags[tag])
            if problem:
                findings.append(
                    _finding(
                        f"timestamp_{problem}",
                        "inconsistent",
                        "low",
                        f"{label} is structurally invalid ({problem}): {_text(tags[tag])!r}.",
                        {"field": label, "raw": _text(tags[tag])},
                    )
                )
                continue
            parsed[label] = value
            if value > now + _FUTURE_TOLERANCE:
                findings.append(
                    _finding(
                        "timestamp_in_future",
                        "suspicious",
                        "low",
                        f"{label} ({value.isoformat()}) is later than the current time.",
                        {"field": label, "value": value.isoformat()},
                    )
                )
        original = parsed.get("DateTimeOriginal")
        modified = parsed.get("DateTime")
        if original and modified and original > modified:
            findings.append(
                _finding(
                    "timestamp_order_inconsistent",
                    "inconsistent",
                    "low",
                    "DateTimeOriginal is later than DateTime (modification time).",
                    {"original": original.isoformat(), "modified": modified.isoformat()},
                )
            )

    def _check_orientation(self, findings: list[dict[str, Any]], orientation: Any) -> None:
        if orientation is not None and (not isinstance(orientation, int) or not 1 <= orientation <= 8):
            findings.append(
                _finding(
                    "orientation_invalid",
                    "inconsistent",
                    "low",
                    f"EXIF orientation {orientation!r} is outside the valid range 1-8.",
                    _sanitize(orientation),
                )
            )

    def _check_gps(
        self, findings: list[dict[str, Any]], block: dict[str, Any], gps_ifd: dict[int, Any]
    ) -> None:
        if not gps_ifd:
            return
        named = {GPSTAGS.get(k, str(k)): v for k, v in gps_ifd.items()}
        lat = _dms_to_decimal(named.get("GPSLatitude"), named.get("GPSLatitudeRef"))
        lon = _dms_to_decimal(named.get("GPSLongitude"), named.get("GPSLongitudeRef"))
        if lat is None or lon is None:
            findings.append(
                _finding("gps_malformed", "inconsistent", "low", "GPS tags are present but could not be decoded.")
            )
            return
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            findings.append(
                _finding(
                    "gps_out_of_range",
                    "inconsistent",
                    "low",
                    f"GPS coordinates ({lat:.5f}, {lon:.5f}) are outside valid ranges.",
                )
            )
            return
        block["gps"] = {"latitude": lat, "longitude": lon}
        findings.append(
            _finding("gps_present", "present", "info", "GPS coordinates recorded.", {"latitude": lat, "longitude": lon})
        )

    def _check_exif_dimensions(
        self,
        findings: list[dict[str, Any]],
        image: Image.Image,
        tags: dict[int, Any],
        orientation: Any,
    ) -> None:
        declared_w, declared_h = tags.get(_TAG_PIXEL_X), tags.get(_TAG_PIXEL_Y)
        if not isinstance(declared_w, int) or not isinstance(declared_h, int):
            return
        width, height = image.size
        if (declared_w, declared_h) in ((width, height), (height, width)):
            return
        findings.append(
            _finding(
                "exif_dimensions_mismatch",
                "inconsistent",
                "low",
                f"EXIF records {declared_w}x{declared_h} but the image is {width}x{height}; "
                "this commonly results from resizing or cropping after capture.",
                {"exif": [declared_w, declared_h], "actual": [width, height]},
            )
        )

    def _read_text_chunks(
        self, image: Image.Image, findings: list[dict[str, Any]]
    ) -> dict[str, Any]:
        chunks: dict[str, Any] = {}
        if image.format != "PNG":
            return chunks
        for key in _PNG_TEXT_KEYS:
            value = image.info.get(key)
            if value is None:
                continue
            text = _text(value)
            chunks[key] = text
            if not text:
                continue
            if key == "parameters" and _A1111_SIGNATURE.search(str(value)):
                findings.append(
                    _finding(
                        "generation_parameters_chunk",
                        "suspicious",
                        "medium",
                        "PNG text chunk contains Stable Diffusion-style generation parameters.",
                        key,
                    )
                )
            elif key in ("prompt", "workflow") and str(value).lstrip().startswith("{"):
                findings.append(
                    _finding(
                        "generation_workflow_chunk",
                        "suspicious",
                        "medium",
                        f"PNG text chunk {key!r} contains a node-graph workflow as written by "
                        "image-generation UIs.",
                        key,
                    )
                )
            elif _GENERATOR_PATTERN.search(str(value)):
                findings.append(
                    _finding(
                        "generator_text_marker",
                        "suspicious",
                        "medium",
                        f"PNG text chunk {key!r} mentions an image-generation tool.",
                        key,
                    )
                )
        return chunks

    @staticmethod
    def _overall_status(exif_status: str, findings: list[dict[str, Any]]) -> str:
        categories = {f["category"] for f in findings}
        if "inconsistent" in categories:
            return "inconsistent"
        if "suspicious" in categories:
            return "suspicious"
        if exif_status in ("present", "absent", "unavailable"):
            return exif_status
        return "unavailable"
