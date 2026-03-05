"""Tests for ImageToolkit utilities."""

from io import BytesIO

import numpy as np
from PIL import Image

from ocr_service.modules.image_toolkit import ImageToolkit


def test_upscale_for_ocr_does_nothing_on_large_image():
    """Test that upscaling does nothing for already large images."""
    img = np.zeros((3000, 3000, 3), dtype=np.uint8)
    out = ImageToolkit.upscale_for_ocr(
        img, max_upscale_factor=2.0, max_long_side_px=3000
    )
    assert out.shape == img.shape


def test_upscale_for_ocr_scales_small_image():
    """Test that upscaling increases size for small images and enforces cap."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    out = ImageToolkit.upscale_for_ocr(
        img, max_upscale_factor=2.0, max_long_side_px=1000
    )
    assert out.shape[0] > img.shape[0] and out.shape[1] > img.shape[1]

    assert max(out.shape[:2]) <= 1000


def test_decode_image_uses_pillow_fallback_when_cv2_imdecode_fails(monkeypatch):
    """Fallback decoder should work when OpenCV cannot decode raw bytes."""
    pil_img = Image.new("RGB", (32, 16), color="white")
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    payload = buf.getvalue()

    monkeypatch.setattr(
        "ocr_service.modules.image_toolkit.cv2.imdecode",
        lambda *_: None,
    )

    decoded = ImageToolkit.decode_image(payload)
    assert decoded is not None
    assert decoded.shape[0] == 16 and decoded.shape[1] == 32


def test_decode_image_uses_pillow_fallback_when_cv2_raises(monkeypatch):
    """Fallback decoder should also work when OpenCV raises an exception."""
    pil_img = Image.new("RGB", (32, 16), color="white")
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    payload = buf.getvalue()

    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated cv2 failure")

    monkeypatch.setattr("ocr_service.modules.image_toolkit.cv2.imdecode", _raise)

    decoded = ImageToolkit.decode_image(payload)
    assert decoded is not None
    assert decoded.shape[0] == 16 and decoded.shape[1] == 32
