"""Cryptographic and perceptual hashing (kept strictly separate)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_CHUNK = 1024 * 1024
_PHASH_SIZE = 8
_PHASH_HIGHFREQ = 4


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    matrix = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    matrix[0, :] *= 1 / np.sqrt(2)
    return matrix * np.sqrt(2 / n)


def perceptual_hash(image: Image.Image) -> dict[str, Any]:
    """Compute a 64-bit DCT perceptual hash (pHash).

    Resizes to 32x32 grayscale, takes the 8x8 low-frequency DCT block, and
    thresholds against its median. Visually similar images give hashes with a
    small Hamming distance; it is unrelated to the SHA-256 file hash.
    """
    size = _PHASH_SIZE * _PHASH_HIGHFREQ
    gray = image.convert("L").resize((size, size), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    matrix = _dct_matrix(size)
    dct = matrix @ pixels @ matrix.T
    low = dct[:_PHASH_SIZE, :_PHASH_SIZE]
    bits = (low > np.median(low)).flatten()
    value = int("".join("1" if b else "0" for b in bits), 2)
    return {
        "algorithm": "phash-dct",
        "hash_size": _PHASH_SIZE,
        "hash": f"{value:0{_PHASH_SIZE * _PHASH_SIZE // 4}x}",
    }


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Hamming distance between two hex perceptual hashes."""
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")
