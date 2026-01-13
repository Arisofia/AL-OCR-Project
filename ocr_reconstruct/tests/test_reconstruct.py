import os
import cv2
from modules.reconstruct import depixelate_naive


def test_depixelate_naive():
    path = os.path.join(os.path.dirname(__file__), "data", "sample_pixelated.png")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    out = depixelate_naive(img)
    assert out is not None
    assert out.shape[0] >= img.shape[0]
