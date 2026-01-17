import os

import cv2

from ocr_reconstruct.modules.enhance import adaptive_threshold, sharpen, to_gray


def test_sharpen_and_threshold():
    path = os.path.join(os.path.dirname(__file__), "data", "sample_clean.png")
    img = cv2.imread(path)
    gray = to_gray(img)
    s = sharpen(gray)
    th = adaptive_threshold(s)
    assert th is not None
    # Ensure threshold yields binary image
    unique_vals = set(th.flatten())
    assert unique_vals.issubset({0, 255})
