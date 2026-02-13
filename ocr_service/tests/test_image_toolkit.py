"""Tests for ImageToolkit utilities."""
import numpy as np

from ocr_service.modules.image_toolkit import ImageToolkit


def test_upscale_for_ocr_does_nothing_on_large_image():
    """Test that upscaling does nothing for already large images."""
    img = np.zeros((3000, 3000, 3), dtype=np.uint8)
    out = ImageToolkit.upscale_for_ocr(img, max_upscale_factor=2.0, max_long_side_px=3000)
    assert out.shape == img.shape  # Already at cap


def test_upscale_for_ocr_scales_small_image():
    """Test that upscaling increases size for small images and enforces cap."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    out = ImageToolkit.upscale_for_ocr(img, max_upscale_factor=2.0, max_long_side_px=1000)
    assert out.shape[0] > img.shape[0] and out.shape[1] > img.shape[1]
    assert max(out.shape[:2]) <= 1000  # Cap enforced
