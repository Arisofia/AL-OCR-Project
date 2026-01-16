"""
Reconstruction utilities: naive depixelation, inpainting, deblur heuristics.

Note: These are intentionally conservative heuristics for research;
full recovery is not guaranteed.
"""

from typing import Optional

import cv2
import numpy as np


class PixelReconstructor:
    """
    Implements logic to recover obscured pixels and remove overlays.
    """

    @staticmethod
    def remove_color_overlay(image: np.ndarray) -> np.ndarray:
        """
        Uses K-means clustering to identify and subtract dominant color overlays.
        Identifies background, text, and overlay layers.
        """
        if len(image.shape) < 3:
            return image

        # Downsample for faster clustering
        small_img = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        pixels = small_img.reshape(-1, 3).astype(np.float32)

        # 3 clusters: Background, Text, and Overlay (e.g., highlighter)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(
            pixels,
            3,
            np.array([], dtype=np.int32),
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS,
        )

        # Find which center is likely the overlay (usually bright but not pure white)
        # Background is typically the brightest
        brightness = np.sum(centers, axis=1)
        bg_idx = np.argmax(brightness)
        text_idx = np.argmin(brightness)
        overlay_idx = 3 - bg_idx - text_idx  # The remaining index

        # Apply the logic to the full image
        full_pixels = image.reshape(-1, 3).astype(np.float32)
        # We can create a mask for the overlay color
        overlay_color = centers[overlay_idx]  # type: ignore

        # Calculate distance of each pixel to the overlay color
        dist = np.linalg.norm(full_pixels - overlay_color, axis=1)
        mask = (dist < 50).reshape(image.shape[:2]).astype(np.uint8) * 255

        # Dilation of mask to cover edges
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        # Inpaint the overlay areas with background color
        bg_color = centers[bg_idx].astype(np.uint8).tolist()  # type: ignore
        result = image.copy()
        result[mask > 0] = bg_color

        return result

    @staticmethod
    def inpaint_text(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Uses Navier-Stokes based inpainting to fill small gaps in characters.
        """
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)

    @staticmethod
    def remove_redactions(image: np.ndarray) -> np.ndarray:
        """
        Attempts to identify black redaction boxes and inpaint them.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image  # type: ignore

        # Black boxes have very low intensity
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)

        # Filter for rectangular shapes (redactions are usually blocks)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(mask)
        for cnt in contours:
            x_coord, y_coord, width, height = cv2.boundingRect(cnt)
            # Filter by size and aspect ratio typical for redactions
            if width > 20 and height > 10:
                cv2.rectangle(
                    final_mask,
                    (x_coord, y_coord),
                    (x_coord + width, y_coord + height),
                    255,
                    -1,
                )

        # Inpaint
        return cv2.inpaint(image, final_mask, 3, cv2.INPAINT_TELEA)

    @staticmethod
    def handle_pixelation(image: np.ndarray, _block_size: int = 8) -> np.ndarray:
        """
        Attempts to reverse pixelation by applying a bilateral filter
        to smooth block boundaries while preserving possible text edges.
        """
        # _block_size is currently unused but kept for interface consistency
        return cv2.bilateralFilter(image, 9, 75, 75)

    @staticmethod
    def depixelate_naive(img_gray: np.ndarray, block: int = 6) -> np.ndarray:
        """
        Naive depixelation: upsample and apply median filter per-block smoothing.
        """
        height, width = img_gray.shape[:2]
        if block <= 1:
            return img_gray
        upscaled = cv2.resize(
            img_gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC
        )
        return cv2.medianBlur(upscaled, 3)


# Backward compatibility
def depixelate_naive(img_gray: np.ndarray, block: int = 6) -> np.ndarray:
    """Wrapper for PixelReconstructor.depixelate_naive"""
    return PixelReconstructor.depixelate_naive(img_gray, block)


def inpaint_bbox(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint using Telea's method given a mask."""
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def deblur_wiener(img: np.ndarray, kernel: Optional[np.ndarray] = None) -> np.ndarray:
    """A very simple Wiener-like deconvolution using a small kernel heuristic."""
    if kernel is None:
        kernel = np.array([[1, 1, 1], [1, 4, 1], [1, 1, 1]], dtype="float32") / 12.0

    img_f = img.astype("float32") / 255.0
    try:
        from scipy import signal  # type: ignore
        from typing import cast

        wiener_filtered = signal.wiener(img_f)
        return cast(np.ndarray, np.clip(wiener_filtered * 255.0, 0, 255)).astype(
            "uint8"
        )
    except (ImportError, Exception):
        # Fallback to a simple sharpening filter
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(img, -1, sharpen_kernel)
