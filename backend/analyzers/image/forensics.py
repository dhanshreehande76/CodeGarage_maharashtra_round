"""Deterministic forensic signals.

Each signal is descriptive evidence. None of them decides authenticity on its
own, and the interpretation bands are uncalibrated heuristics (flagged as such
in the output).
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
from PIL import Image, JpegImagePlugin

from .imaging import downscale

logger = logging.getLogger(__name__)

_STD_LUMA = (
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
)
_SUBSAMPLING = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}

_ELA_TILE = 16
_ELA_MIN_TILES = 4
_ELA_BAND_LOW = 0.5
_ELA_BAND_HIGH = 1.0
_ELA_MIN_MEAN_ERROR = 0.25
_PEAK_PROMINENCE_THRESHOLD = 2.5
_PEAK_NEIGHBOUR_OFFSET = 4
_MIN_FREQ_SIDE = 64
_JPEG_FORMATS = frozenset({"JPEG", "MPO"})


def _f(value: Any, digits: int = 6) -> float | None:
    number = float(value)
    return None if not np.isfinite(number) else round(number, digits)


def _ijg_scaled_sum(quality: int) -> int:
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    return sum(min(255, max(1, int((v * scale + 50) // 100))) for v in _STD_LUMA)


def estimate_jpeg_quality(luma_table: list[int]) -> tuple[int, float]:
    """Estimate IJG quality from a luminance table; returns (quality, relative fit error)."""
    total = sum(luma_table)
    best = min(range(1, 101), key=lambda q: (abs(_ijg_scaled_sum(q) - total), -q))
    error = abs(_ijg_scaled_sum(best) - total) / max(total, 1)
    return best, error


def compression_info(image: Image.Image) -> dict[str, Any]:
    """Describe the compression of the source file where Pillow exposes it."""
    fmt = image.format
    info: dict[str, Any] = {"format": fmt, "jpeg": None, "notes": []}
    if fmt not in _JPEG_FORMATS:
        info["notes"].append(f"JPEG quality is not applicable to {fmt} files.")
        return info

    tables = getattr(image, "quantization", None) or {}
    jpeg: dict[str, Any] = {
        "quantization_table_count": len(tables),
        "estimated_quality": None,
        "quality_fit_relative_error": None,
        "quality_method": "ijg_luminance_table_sum",
        "subsampling": None,
        "progressive": bool(image.info.get("progressive") or image.info.get("progression")),
    }
    luma = tables.get(0)
    if luma:
        quality, error = estimate_jpeg_quality([int(v) for v in luma])
        jpeg["estimated_quality"] = quality
        jpeg["quality_fit_relative_error"] = _f(error)
        if error > 0.05:
            info["notes"].append(
                "Quantization table deviates from the standard IJG scaling; the quality "
                "value is only a rough estimate (custom or camera-specific tables)."
            )
    else:
        info["notes"].append("No quantization table available.")
    try:
        sampling = JpegImagePlugin.get_sampling(image)
        jpeg["subsampling"] = _SUBSAMPLING.get(sampling)
    except Exception:
        jpeg["subsampling"] = None
    info["jpeg"] = jpeg
    return info


def error_level_analysis(
    rgb: Image.Image, source_format: str | None, quality: int, max_pixels: int
) -> dict[str, Any]:
    """Recompress as JPEG and measure the per-tile residual (Error Level Analysis)."""
    notes: list[str] = []
    work = rgb
    if rgb.width * rgb.height > max_pixels:
        scale = (max_pixels / (rgb.width * rgb.height)) ** 0.5
        work = rgb.resize(
            (max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))), Image.BILINEAR
        )
        notes.append(f"Image downscaled to {work.width}x{work.height} before ELA to bound cost.")

    buffer = io.BytesIO()
    work.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        recompressed_rgb = recompressed.convert("RGB")
        resaved = np.asarray(recompressed_rgb, dtype=np.int16)
    original = np.asarray(work, dtype=np.int16)
    residual = np.abs(original - resaved).astype(np.float32).mean(axis=2)

    height, width = residual.shape
    rows, cols = height // _ELA_TILE, width // _ELA_TILE
    result: dict[str, Any] = {
        "recompression_quality": quality,
        "mean_error": _f(residual.mean()),
        "std_error": _f(residual.std()),
        "p99_error": _f(np.percentile(residual, 99)),
        "max_error": _f(residual.max()),
        "tile_size": _ELA_TILE,
        "tile_grid": [rows, cols],
        "tile_mean_cv": None,
        "hot_tile_fraction": None,
        "applicability": "applicable" if source_format in _JPEG_FORMATS else "limited",
        "bands_calibrated": False,
        "notes": notes,
        "signal": None,
    }
    if source_format not in _JPEG_FORMATS:
        notes.append(
            "Source is not JPEG; ELA is built around JPEG recompression history, so the "
            "values are of limited forensic meaning."
        )
    if rows < _ELA_MIN_TILES or cols < _ELA_MIN_TILES:
        notes.append("Image too small for tile-level ELA statistics.")
        return result

    tiles = residual[: rows * _ELA_TILE, : cols * _ELA_TILE]
    tile_means = tiles.reshape(rows, _ELA_TILE, cols, _ELA_TILE).mean(axis=(1, 3))
    mean_of_tiles = float(tile_means.mean())
    std_of_tiles = float(tile_means.std())
    cv = std_of_tiles / mean_of_tiles if mean_of_tiles > 1e-6 else 0.0
    hot = float((tile_means > mean_of_tiles + 3 * std_of_tiles).mean()) if std_of_tiles > 0 else 0.0
    result["tile_mean_cv"] = _f(cv)
    result["hot_tile_fraction"] = _f(hot)

    if mean_of_tiles < _ELA_MIN_MEAN_ERROR:
        band, text = "negligible", (
            "negligible recompression error; the image appears to already be saved at about "
            "this JPEG quality, so tile variation is not meaningful"
        )
    elif cv < _ELA_BAND_LOW:
        band, text = "low", "low variation in recompression error across the image"
    elif cv < _ELA_BAND_HIGH:
        band, text = "moderate", "moderate localized variation in recompression error"
    else:
        band, text = "high", (
            "pronounced localized variation in recompression error; regions may differ in "
            "compression history (e.g. editing, resampling, or naturally flat versus textured areas)"
        )
    result["band"] = band
    result["signal"] = {"signal": "ela_variation", "value": _f(cv), "interpretation": text}
    return result


def image_statistics(rgb: Image.Image, max_side: int) -> dict[str, Any]:
    """Basic colour/sharpness statistics computed on a bounded-size copy."""
    work = downscale(rgb, max_side)
    arr = np.asarray(work, dtype=np.float32)
    gray_u8 = np.asarray(work.convert("L"))
    gray = gray_u8.astype(np.float32)

    histogram = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    probs = histogram[histogram > 0] / histogram.sum()
    entropy = float(-(probs * np.log2(probs)).sum())

    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        lap = (
            4 * gray[1:-1, 1:-1]
            - gray[:-2, 1:-1]
            - gray[2:, 1:-1]
            - gray[1:-1, :-2]
            - gray[1:-1, 2:]
        )
        laplacian_var: float | None = _f(lap.var())
    else:
        laplacian_var = None

    rgb_u32 = np.asarray(work, dtype=np.uint32)
    packed = (rgb_u32[..., 0] << 16) | (rgb_u32[..., 1] << 8) | rgb_u32[..., 2]
    return {
        "analysis_resolution": [work.width, work.height],
        "channel_mean_rgb": [_f(v, 4) for v in arr.mean(axis=(0, 1))],
        "channel_std_rgb": [_f(v, 4) for v in arr.std(axis=(0, 1))],
        "luminance_mean": _f(gray.mean(), 4),
        "luminance_std": _f(gray.std(), 4),
        "luminance_entropy_bits": _f(entropy, 4),
        "laplacian_variance": laplacian_var,
        "clipped_black_fraction": _f((gray_u8 == 0).mean()),
        "clipped_white_fraction": _f((gray_u8 == 255).mean()),
        "unique_colors_at_analysis_resolution": int(np.unique(packed).size),
    }


def frequency_analysis(rgb: Image.Image, crop_size: int) -> dict[str, Any]:
    """Radially averaged power-spectrum statistics from a centre crop."""
    width, height = rgb.size
    side = min(crop_size, width, height)
    if side < _MIN_FREQ_SIDE:
        return {
            "available": False,
            "reason": f"Image shorter side is below {_MIN_FREQ_SIDE}px.",
            "signal": None,
        }
    side -= side % 2
    left, top = (width - side) // 2, (height - side) // 2
    crop = rgb.crop((left, top, left + side, top + side)).convert("L")
    data = np.asarray(crop, dtype=np.float64)

    window = np.outer(np.hanning(side), np.hanning(side))
    spectrum = np.fft.fftshift(np.fft.fft2((data - data.mean()) * window))
    power = np.abs(spectrum) ** 2

    yy, xx = np.indices(power.shape)
    radius = np.hypot(yy - side // 2, xx - side // 2)
    r_max = side // 2
    mask = (radius >= 1) & (radius < r_max)
    bins = np.floor(radius[mask]).astype(int)
    values = power[mask]
    counts = np.bincount(bins, minlength=r_max)
    profile = np.bincount(bins, weights=values, minlength=r_max)
    valid = counts > 0
    profile[valid] /= counts[valid]

    total = values.sum()
    if total <= 0:
        return {"available": False, "reason": "Image has no spectral energy (constant image).", "signal": None}
    high_ratio = float(values[radius[mask] / r_max >= 0.5].sum() / total)

    freqs = np.arange(r_max)
    fit_mask = valid & (freqs >= 2) & (profile > 0)
    slope: float | None = None
    if fit_mask.sum() >= 3:
        slope = float(np.polyfit(np.log10(freqs[fit_mask]), np.log10(profile[fit_mask]), 1)[0])

    log_power = np.log10(power + 1e-12)
    offset = _PEAK_NEIGHBOUR_OFFSET
    neighbours = np.log10(
        (
            np.roll(power, offset, axis=0)
            + np.roll(power, -offset, axis=0)
            + np.roll(power, offset, axis=1)
            + np.roll(power, -offset, axis=1)
        )
        / 4
        + 1e-12
    )
    high_mask = mask & (radius / r_max >= 0.5)
    prominence = float((log_power - neighbours)[high_mask].max()) if high_mask.any() else 0.0

    if prominence >= _PEAK_PROMINENCE_THRESHOLD:
        interpretation = (
            "isolated periodic peaks in the high-frequency spectrum; such patterns can arise "
            "from resampling, JPEG block grids or generator upsampling, and also from "
            "legitimate processing"
        )
    else:
        interpretation = "no isolated high-frequency spectral peaks relative to neighbouring frequencies"

    return {
        "available": True,
        "crop_size": side,
        "high_frequency_energy_ratio": _f(high_ratio),
        "spectral_slope": _f(slope) if slope is not None else None,
        "spectral_peak_prominence_log10": _f(prominence),
        "peak_prominence_threshold": _PEAK_PROMINENCE_THRESHOLD,
        "bands_calibrated": False,
        "signal": {
            "signal": "spectral_peak_prominence",
            "value": _f(prominence),
            "interpretation": interpretation,
        },
    }
