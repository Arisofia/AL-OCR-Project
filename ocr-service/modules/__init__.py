"""
Entry point for ocr-service modules.
"""

from .advanced_recon import AdvancedPixelReconstructor
from .ocr_engine import IterativeOCREngine

try:
    from ocr_reconstruct.modules.enhance import ImageEnhancer
    from ocr_reconstruct.modules.reconstruct import PixelReconstructor
except ImportError:
    # Provide lightweight fallbacks so the ocr-service can run
    # without the optional ocr_reconstruct package.
    # pylint: disable=no-member
    import cv2
    import numpy as _np

    class ImageEnhancer:  # type: ignore[no-redef]
        """Minimal ImageEnhancer fallback used for tests and lightweight deployments."""

        def sharpen(self, img: _np.ndarray) -> _np.ndarray:
            """Simple sharpening pass."""
            kernel = _np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=_np.float32)
            return cv2.filter2D(img, -1, kernel)

        def apply_threshold(self, img: _np.ndarray) -> _np.ndarray:
            """Simple thresholding pass."""
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img  # type: ignore
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            return thresh

    class PixelReconstructor:  # type: ignore[no-redef]
        """Very small stub of PixelReconstructor to satisfy imports in tests."""

        def reconstruct(self, img: _np.ndarray) -> _np.ndarray:
            """Passthrough reconstruction stub."""
            return img


from .learning_engine import LearningEngine
from .processor import OCRProcessor

__all__ = [
    "IterativeOCREngine",
    "AdvancedPixelReconstructor",
    "ImageEnhancer",
    "PixelReconstructor",
    "LearningEngine",
    "OCRProcessor",
]
