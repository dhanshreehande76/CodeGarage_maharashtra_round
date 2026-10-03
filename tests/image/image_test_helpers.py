"""Deterministic image builders shared by the image tests."""

from __future__ import annotations

import numpy as np
from PIL import Image


def make_pixels(width: int = 160, height: int = 120, seed: int = 0) -> Image.Image:
    """Deterministic smooth gradient plus mild noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack(
        [x * 255 / width, y * 255 / height, (x + y) * 255 / (width + height)], axis=-1
    )
    noisy = base + rng.normal(0, 6, base.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype("uint8"))
