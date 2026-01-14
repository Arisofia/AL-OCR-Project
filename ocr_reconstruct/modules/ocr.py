"""
Thin wrapper around pytesseract to centralize config and calls.
"""

import pytesseract
import cv2
import numpy as np


def image_to_text(
    img: np.ndarray, lang: str = "eng", psm: int = 6, oem: int = 3
) -> str:
    """
    Converts an image (numpy array) to text using Tesseract.
    Accepts both grayscale and color images.
    """
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    config = f"--oem {oem} --psm {psm}"
    text = pytesseract.image_to_string(img_gray, lang=lang, config=config)
    return text.strip()
