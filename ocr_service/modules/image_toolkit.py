"""
Utility toolkit for image processing operations.
"""

import asyncio
import base64
import binascii
import logging
from io import BytesIO
from typing import Any, Optional

# pylint: disable=no-member
import cv2
import numpy as np
from PIL import Image

__all__ = ["ImageToolkit"]

logger = logging.getLogger("ocr-service.image-toolkit")


class ImageToolkitError(Exception):
    """Custom exception for image toolkit errors."""


class ImageToolkit:
    @staticmethod
    def prepare_image_bytes(data: Any) -> Optional[bytes]:
        """
        Universal handler for various input formats.
        Converts base64 strings or raw data into bytes.
        """
        if data is None:
            return None

        if isinstance(data, bytes):
            return data

        if isinstance(data, str):
            try:
                # Check for common base64 data URL prefixes
                if data.startswith("data:image"):
                    data = data.split(",")[-1]
                return base64.b64decode(data)
            except (binascii.Error, TypeError) as e:
                logger.error("Failed to decode base64 image data: %s", e)
                raise ImageToolkitError("Invalid base64 image data") from e

        return None

    @staticmethod
    def decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Decodes raw image bytes into an OpenCV-compatible numpy array.
        Synchronous helper for use within threads.
        """
        try:
            img: Optional[np.ndarray] = None
            try:
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as cv_exc:
                logger.warning(
                    "cv2.imdecode raised an exception; attempting Pillow fallback: %s",
                    cv_exc,
                )

            if img is None:
                logger.warning(
                    "cv2.imdecode returned None; attempting Pillow fallback decoder"
                )
                try:
                    with Image.open(BytesIO(image_bytes)) as pil_img:
                        rgb = pil_img.convert("RGB")
                        # Avoid cv2 color conversion in fallback path because some
                        # Lambda/OpenCV builds fail ndarray type checks.
                        img = np.array(rgb)[:, :, ::-1].copy()
                except Exception as pil_exc:
                    logger.error(
                        "Pillow fallback decoder failed after cv2 failure: %s",
                        pil_exc,
                    )
                    raise ImageToolkitError(
                        "Failed to decode image: cv2 and Pillow decoders failed"
                    ) from pil_exc
            return img
        except Exception as e:
            logger.error("Error decoding image: %s", e)
            raise ImageToolkitError(f"Error decoding image: {e}") from e

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

    @staticmethod
    def upscale_for_ocr(
        img: np.ndarray,
        max_upscale_factor: float,
        max_long_side_px: int,
    ) -> np.ndarray:
        """
        Upscale an image to improve OCR readability while respecting resource limits.

        This function increases the effective resolution of the input image when it is
        too small, but caps both the upscale factor and the maximum long-edge size
        to avoid excessive memory and latency costs.

        Args:
            img: The input image as a NumPy array (H x W x C or H x W).
            max_upscale_factor: The maximum factor by which to enlarge the image.
            max_long_side_px: The maximum allowed size (in pixels) of the image's
                longer edge after upscaling.

        Returns:
            The upscaled (or original) image as a NumPy array.
        """
        height, width = img.shape[:2]
        long_side = max(height, width)

        # If the image is already large enough, do nothing.
        if long_side >= max_long_side_px:
            return img

        # Heuristic: upsample small images more aggressively.
        # Example:
        #   < 800px  -> try 2x
        #   < 1500px -> try 1.5x
        #   otherwise -> no upscaling
        if long_side < 800:
            desired_scale = 2.0
        elif long_side < 1500:
            desired_scale = 1.5
        else:
            desired_scale = 1.0

        # Enforce configured max upscale factor.
        scale = min(desired_scale, max_upscale_factor)
        if scale <= 1.0:
            return img

        # Compute new size but cap by max_long_side_px.
        new_width = int(width * scale)
        new_height = int(height * scale)

        long_after = max(new_height, new_width)
        if long_after > max_long_side_px:
            # Adjust scale so that the long side equals max_long_side_px.
            cap_scale = max_long_side_px / float(long_side)
            new_width = int(width * cap_scale)
            new_height = int(height * cap_scale)

        # Use bicubic interpolation for better text detail enlargement.
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
