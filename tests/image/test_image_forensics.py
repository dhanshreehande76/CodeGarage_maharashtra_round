from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from backend.analyzers.image import forensics
from image_test_helpers import make_pixels


@pytest.mark.parametrize("quality", [30, 50, 75, 90, 95])
def test_jpeg_quality_estimate_matches_pillow_setting(tmp_path, quality):
    path = tmp_path / "q.jpg"
    make_pixels().save(path, quality=quality)
    with Image.open(path) as image:
        info = forensics.compression_info(image)
    assert info["jpeg"]["estimated_quality"] == quality
    assert info["jpeg"]["subsampling"] == "4:2:0"


def test_non_jpeg_has_no_jpeg_block(png_path):
    with Image.open(png_path) as image:
        info = forensics.compression_info(image)
    assert info["jpeg"] is None and info["notes"]


def test_ela_signal_shape_and_cautious_wording(jpeg_path):
    with Image.open(jpeg_path) as image:
        rgb = image.convert("RGB")
        ela = forensics.error_level_analysis(rgb, "JPEG", 90, 16_000_000)
    assert ela["applicability"] == "applicable"
    assert ela["signal"]["signal"] == "ela_variation"
    assert isinstance(ela["signal"]["value"], float)
    text = ela["signal"]["interpretation"].lower()
    assert "fake" not in text and "ai-generated" not in text and "proves" not in text
    assert ela["bands_calibrated"] is False


def test_ela_is_deterministic(jpeg_path):
    with Image.open(jpeg_path) as image:
        rgb = image.convert("RGB")
    a = forensics.error_level_analysis(rgb, "JPEG", 90, 16_000_000)
    b = forensics.error_level_analysis(rgb, "JPEG", 90, 16_000_000)
    assert a == b


def test_ela_limited_for_png_and_small_images(png_path):
    with Image.open(png_path) as image:
        rgb = image.convert("RGB")
    assert forensics.error_level_analysis(rgb, "PNG", 90, 16_000_000)["applicability"] == "limited"
    tiny = forensics.error_level_analysis(Image.new("RGB", (20, 20), "gray"), "JPEG", 90, 16_000_000)
    assert tiny["signal"] is None and tiny["notes"]


def test_ela_localized_edit_raises_variation(tmp_path):
    base = make_pixels(256, 256)
    path = tmp_path / "base.jpg"
    base.save(path, quality=70)
    with Image.open(path) as clean:
        clean_rgb = clean.convert("RGB")
    edited = clean_rgb.copy()
    patch = np.random.default_rng(5).integers(0, 255, (48, 48, 3), dtype=np.uint8)
    edited.paste(Image.fromarray(patch), (100, 100))
    clean_cv = forensics.error_level_analysis(clean_rgb, "JPEG", 90, 16_000_000)["tile_mean_cv"]
    edited_cv = forensics.error_level_analysis(edited, "JPEG", 90, 16_000_000)["tile_mean_cv"]
    assert edited_cv > clean_cv


def test_image_statistics(jpeg_path):
    with Image.open(jpeg_path) as image:
        stats = forensics.image_statistics(image.convert("RGB"), 1024)
    assert stats["analysis_resolution"] == [160, 120]
    assert len(stats["channel_mean_rgb"]) == 3
    assert 0 <= stats["luminance_entropy_bits"] <= 8
    assert stats["laplacian_variance"] >= 0


def test_image_statistics_downscales_large_input():
    stats = forensics.image_statistics(Image.new("RGB", (4000, 2000), "white"), 1024)
    assert max(stats["analysis_resolution"]) == 1024
    assert stats["clipped_white_fraction"] == 1.0


def test_frequency_detects_injected_periodic_pattern():
    rng = np.random.default_rng(0)
    noise = rng.normal(128, 20, (256, 256))
    x = np.arange(256)
    pattern = 40 * np.sin(2 * np.pi * x / 3.0)[None, :]
    plain = Image.fromarray(np.clip(noise, 0, 255).astype("uint8"))
    periodic = Image.fromarray(np.clip(noise + pattern, 0, 255).astype("uint8"))
    plain_result = forensics.frequency_analysis(plain.convert("RGB"), 256)
    result = forensics.frequency_analysis(periodic.convert("RGB"), 256)
    key, threshold = "spectral_peak_prominence_log10", result["peak_prominence_threshold"]
    assert plain_result[key] < threshold
    assert result[key] >= threshold


def test_frequency_unavailable_for_tiny_or_constant():
    assert forensics.frequency_analysis(Image.new("RGB", (32, 32)), 512)["available"] is False
    assert forensics.frequency_analysis(Image.new("RGB", (128, 128), "gray"), 512)["available"] is False


def test_ela_negligible_residual_is_not_reported_as_variation(tmp_path):
    path = tmp_path / "q90.jpg"
    make_pixels(640, 480, seed=7).save(path, quality=90)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    ela = forensics.error_level_analysis(rgb, "JPEG", 90, 16_000_000)
    assert ela["band"] == "negligible"
    assert "negligible" in ela["signal"]["interpretation"]
