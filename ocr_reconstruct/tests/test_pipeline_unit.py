from unittest import mock

import cv2
import numpy as np

from ocr_reconstruct.modules.pipeline import IterativeOCR


def test_process_bytes_invalid_input():
    """Test process_bytes with empty or invalid byte stream."""
    worker = IterativeOCR(iterations=1)
    text, final_img, meta = worker.process_bytes(b"not an image")

    assert text == ""
    assert final_img is None
    assert "error" in meta
    assert meta["error"] == "Invalid byte stream payload"


@mock.patch("ocr_reconstruct.modules.ocr.pytesseract.image_to_string")
def test_process_bytes_mocked(mock_tesseract):
    """Test process_bytes with a mocked Tesseract call to ensure logic flow."""
    mock_tesseract.return_value = "Mocked OCR Result"

    # Create a simple 10x10 white square image bytes
    img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    _, img_encoded = cv2.imencode(".png", img)
    img_bytes = img_encoded.tobytes()

    worker = IterativeOCR(iterations=1)
    text, final_img, meta = worker.process_bytes(img_bytes)

    assert text == "Mocked OCR Result"
    assert final_img is not None
    assert "iterations" in meta
    assert len(meta["iterations"]) > 0
    mock_tesseract.assert_called()
