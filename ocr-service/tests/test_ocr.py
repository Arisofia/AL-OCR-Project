import pytest
import numpy as np
import cv2
from modules.enhance import ImageEnhancer

def test_ocr_engine_invalid_input():
    from modules.ocr_engine import IterativeOCREngine
    engine = IterativeOCREngine()
    
    # Test empty input
    result = engine.process_image(b"")
    assert "error" in result
    assert result["error"] == "Empty image content"
    
    # Test large input
    large_input = b"0" * (11 * 1024 * 1024)
    result = engine.process_image(large_input)
    assert "error" in result
    assert "exceeds 10MB" in result["error"]

def test_image_enhancer_sharpen():
    # Create a blurred image to see sharpening effect
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (25, 25), (75, 75), (255, 255, 255), -1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    enhancer = ImageEnhancer()
    sharpened = enhancer.sharpen(img)
    
    assert sharpened.shape == img.shape
    assert np.any(sharpened != img)

def test_image_enhancer_threshold():
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.putText(img, "TEST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    enhancer = ImageEnhancer()
    thresh = enhancer.apply_threshold(img)
    
    assert len(thresh.shape) == 2
    assert np.max(thresh) == 255
    assert np.min(thresh) == 0
