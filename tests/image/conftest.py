"""Shared fixtures for the image analyzer tests (no network, no model download)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from image_test_helpers import make_pixels  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "integration: needs the real pretrained model (cached locally)"
    )


@pytest.fixture
def jpeg_path(tmp_path: Path) -> Path:
    path = tmp_path / "photo.jpg"
    make_pixels().save(path, quality=90)
    return path


@pytest.fixture
def png_path(tmp_path: Path) -> Path:
    path = tmp_path / "picture.png"
    make_pixels(seed=1).save(path)
    return path


@pytest.fixture
def exif_jpeg_path(tmp_path: Path) -> Path:
    exif = Image.Exif()
    exif[0x010F] = "TestCam"
    exif[0x0110] = "Model X"
    exif[0x0131] = "TestFirmware 1.0"
    exif[0x0132] = "2024:05:01 10:00:00"
    exif[0x0112] = 1
    exif_ifd = exif.get_ifd(0x8769)
    exif_ifd[0x9003] = "2024:05:01 09:59:59"
    exif_ifd[0x9004] = "2024:05:01 09:59:59"
    gps = exif.get_ifd(0x8825)
    gps[1] = "N"
    gps[2] = (18.0, 31.0, 12.0)
    gps[3] = "E"
    gps[4] = (73.0, 51.0, 36.0)
    path = tmp_path / "camera.jpg"
    make_pixels().save(path, quality=92, exif=exif)
    return path


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny randomly initialised ViT with the same architecture class and
    single-logit head as the real detector. Used only to exercise the code path;
    its outputs carry no authenticity meaning."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

    directory = tmp_path_factory.mktemp("tiny_vit")
    config = ViTConfig(
        image_size=32,
        patch_size=16,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=1,
    )
    torch.manual_seed(0)
    ViTForImageClassification(config).save_pretrained(directory)
    ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(directory)
    return directory
