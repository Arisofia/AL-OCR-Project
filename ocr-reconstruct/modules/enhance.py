"""Image enhancement helpers: grayscale, sharpening, denoising, thresholding."""
import cv2
import numpy as np


def to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def sharpen(img_gray):
    """Apply a small sharpening kernel."""
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img_gray, -1, kernel)


def denoise(img_gray):
    return cv2.fastNlMeansDenoising(img_gray, None, h=10)


def adaptive_threshold(img_gray):
    # Use Otsu's threshold for general cases
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def upscale_and_smooth(img_gray, scale=2):
    # Upsample then apply median blur to reduce pixelation
    h, w = img_gray.shape[:2]
    up = cv2.resize(img_gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    smooth = cv2.medianBlur(up, 3)
    return smooth
