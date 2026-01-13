"""Reconstruction utilities: naive depixelation, inpainting, deblur heuristics.

Note: These are intentionally conservative heuristics for research; full recovery is not guaranteed.
"""
import cv2
import numpy as np


def depixelate_naive(img_gray, block=6):
    """Naive depixelation: upsample and apply median filter per-block smoothing."""
    h, w = img_gray.shape[:2]
    # If block is 1, nothing to do
    if block <= 1:
        return img_gray

    # Resize using INTER_CUBIC
    up = cv2.resize(img_gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    out = cv2.medianBlur(up, 3)
    return out


def inpaint_bbox(img, mask):
    """Inpaint using Telea's method given a mask (non-zero areas are inpainted)."""
    inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return inpainted


def deblur_wiener(img, kernel=None):
    """A very simple Wiener-like deconvolution using a small kernel heuristic.
    This is a lightweight approximation and will not replace full deconvolution libraries.
    """
    if kernel is None:
        kernel = np.array([[1,1,1],[1,4,1],[1,1,1]], dtype='float32')/12.0

    # Convert to float
    img_f = img.astype('float32') / 255.0
    # FFT-based deconvolution (naive)
    try:
        import scipy.signal as signal
        deconv = signal.wiener(img_f)
        res = np.clip(deconv*255.0, 0, 255).astype('uint8')
        return res
    except Exception:
        # fallback to sharpening if scipy not available
        return cv2.filter2D(img, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))
