import pytest
import numpy as np
import cv2
from modules.enhance import ImageEnhancer

def test_image_enhancer_sharpen():
    # Create a simple blank image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (25, 25), (75, 75), (255, 255, 255), -1)
    
    enhancer = ImageEnhancer()
    sharpened = enhancer.sharpen(img)
    
    assert sharpened.shape == img.shape
    assert np.any(sharpened != img) # Sharpening should modify the image

def test_image_enhancer_threshold():
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.putText(img, "TEST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    enhancer = ImageEnhancer()
    thresh = enhancer.apply_threshold(img)
    
    assert len(thresh.shape) == 2
    assert np.max(thresh) == 255
    assert np.min(thresh) == 0
