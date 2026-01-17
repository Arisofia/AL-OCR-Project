"""
Utility toolkit for image processing operations.
"""

import asyncio
import logging
from typing import Optional

# pylint: disable=no-member
import cv2
import numpy as np

logger = logging.getLogger("ocr-service.image-toolkit")


class ImageToolkit:
    @staticmethod
    def decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Decodes raw image bytes into an OpenCV-compatible numpy array.
        Synchronous helper for use within threads.
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("Failed to decode image bytes.")
            return img
        except Exception as e:
            logger.error("Error decoding image: %s", e)
            return None

    @staticmethod
    async def decode_image_async(image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Asynchronously decodes image bytes.
        """
        return await asyncio.to_thread(ImageToolkit.decode_image, image_bytes)

    @staticmethod
    def validate_image(image_bytes: bytes, max_size_mb: int = 10) -> Optional[str]:
        """
        Validates image content and size.
        """
        if not image_bytes:
            return "Empty image content"

        if len(image_bytes) > max_size_mb * 1024 * 1024:
            return f"Image size exceeds {max_size_mb}MB limit"

        return None

    @staticmethod
    def prepare_roi(roi: np.ndarray, padding: int = 10) -> np.ndarray:
        """
        Applies padding to a Region of Interest for better OCR performance.
        """
        return cv2.copyMakeBorder(
            roi, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255]
        )

    @staticmethod
    async def enhance_iteration(img: np.ndarray) -> np.ndarray:
        """
        Asynchronous enhancement pass between iterations.
        """

        def _enhance():
            return cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

        return await asyncio.to_thread(_enhance)
