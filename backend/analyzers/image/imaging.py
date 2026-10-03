"""Safe image loading and colour-mode normalisation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import ImageAnalysisError

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = frozenset({"JPEG", "MPO", "PNG", "WEBP", "BMP", "TIFF"})

FORMAT_EXTENSIONS: dict[str, frozenset[str]] = {
    "JPEG": frozenset({".jpg", ".jpeg", ".jpe", ".jfif"}),
    "MPO": frozenset({".jpg", ".jpeg", ".jpe", ".jfif", ".mpo"}),
    "PNG": frozenset({".png"}),
    "WEBP": frozenset({".webp"}),
    "BMP": frozenset({".bmp", ".dib"}),
    "TIFF": frozenset({".tif", ".tiff"}),
}

SUPPORTED_EXTENSIONS = frozenset().union(*FORMAT_EXTENSIONS.values())

_ORIENTATION_TAG = 0x0112
_SIXTEEN_BIT_MODES = frozenset({"I;16", "I;16L", "I;16B", "I;16N", "I"})


def open_image(
    path: Path, *, max_file_size_bytes: int, max_pixels: int
) -> Image.Image:
    """Open and fully decode an image, raising ``ImageAnalysisError`` on any problem."""
    if not path.exists():
        raise ImageAnalysisError("file_not_found", f"File does not exist: {path}", "input")
    if not path.is_file():
        raise ImageAnalysisError("not_a_file", f"Path is not a regular file: {path}", "input")

    size = path.stat().st_size
    if size == 0:
        raise ImageAnalysisError("empty_file", "File is empty.", "input")
    if size > max_file_size_bytes:
        raise ImageAnalysisError(
            "file_too_large",
            f"File is {size} bytes; the limit is {max_file_size_bytes} bytes.",
            "input",
        )

    try:
        image = Image.open(path)
    except UnidentifiedImageError as exc:
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            raise ImageAnalysisError(
                "corrupted_image",
                "File has an image extension but its content cannot be identified as an image.",
                "input",
            ) from exc
        raise ImageAnalysisError(
            "unsupported_file", "File is not a supported image type.", "input"
        ) from exc
    except Image.DecompressionBombError as exc:
        raise ImageAnalysisError("image_too_large", str(exc), "input") from exc
    except (OSError, ValueError, SyntaxError) as exc:
        raise ImageAnalysisError("unreadable_image", f"Image could not be opened: {exc}", "input") from exc

    try:
        if image.format not in SUPPORTED_FORMATS:
            raise ImageAnalysisError(
                "unsupported_file",
                f"Image format {image.format!r} is not supported "
                f"(supported: {', '.join(sorted(SUPPORTED_FORMATS))}).",
                "input",
            )
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ImageAnalysisError("invalid_dimensions", f"Invalid dimensions {width}x{height}.", "input")
        if width * height > max_pixels:
            raise ImageAnalysisError(
                "image_too_large",
                f"Image has {width * height} pixels; the limit is {max_pixels}.",
                "input",
            )
        try:
            image.load()
        except Image.DecompressionBombError as exc:
            raise ImageAnalysisError("image_too_large", str(exc), "input") from exc
        except (OSError, ValueError, SyntaxError, EOFError) as exc:
            raise ImageAnalysisError(
                "corrupted_image", f"Image data is corrupted or truncated: {exc}", "input"
            ) from exc
    except ImageAnalysisError:
        image.close()
        raise
    return image


def to_rgb(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    """Convert any supported mode to 8-bit RGB, flattening transparency onto white."""
    info: dict[str, Any] = {"source_mode": image.mode, "alpha_flattened": False}
    mode = image.mode

    if mode in _SIXTEEN_BIT_MODES:
        arr = np.asarray(image).astype(np.float32)
        if arr.max() > 255:
            arr = arr / 256.0
        gray = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
        return gray.convert("RGB"), info

    has_alpha = mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in image.info)
    if has_alpha:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        info["alpha_flattened"] = True
        return Image.alpha_composite(background, rgba).convert("RGB"), info

    try:
        return image.convert("RGB"), info
    except ValueError as exc:
        raise ImageAnalysisError(
            "unsupported_color_mode", f"Cannot convert mode {mode!r} to RGB: {exc}", "input"
        ) from exc


def prepare_for_model(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    """Apply EXIF orientation and convert to RGB for detector input."""
    transposed = False
    try:
        orientation = image.getexif().get(_ORIENTATION_TAG, 1)
    except Exception:
        orientation = 1
    if orientation in (2, 3, 4, 5, 6, 7, 8):
        fixed = ImageOps.exif_transpose(image)
        if fixed is not None:
            image = fixed
            transposed = True
    rgb, info = to_rgb(image)
    info["exif_transposed"] = transposed
    return rgb, info


def downscale(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the longer side is at most ``max_side`` (deterministic, bilinear)."""
    longest = max(image.size)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(new_size, Image.BILINEAR)
