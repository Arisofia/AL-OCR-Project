"""
Image enhancement helpers: grayscale, sharpening, denoising, thresholding.
Provides the ImageEnhancer class and functional wrappers for image preprocessing.
"""

import cv2
import numpy as np

__all__ = [
    "ImageEnhancer",
    "adaptive_threshold",
    "denoise",
    "sharpen",
    "to_gray",
    "upscale_and_smooth",
]


class ImageEnhancer:
    """
    Handles image preprocessing for OCR enhancement.
    """

    @staticmethod
    def to_gray(img: np.ndarray) -> np.ndarray:
        """
        Converts an image to grayscale if it is in color.
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    @staticmethod
    def sharpen(img_gray: np.ndarray) -> np.ndarray:
        """
        Apply a small sharpening kernel to enhance edges.
        """
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(img_gray, -1, kernel)

    @staticmethod
    def denoise(img_gray: np.ndarray) -> np.ndarray:
        """
        Applies Non-Local Means Denoising to the grayscale image.
        """
        return cv2.fastNlMeansDenoising(img_gray, None, h=10)

    @staticmethod
    def apply_threshold(img_gray: np.ndarray) -> np.ndarray:
        """
        Applies Otsu's thresholding after a Gaussian blur.
        """
        blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        _, thresholded = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return thresholded

    @staticmethod
    def upscale_and_smooth(img_gray: np.ndarray, scale: int = 2) -> np.ndarray:
        """
        Upsamples the image and applies a median blur to reduce pixelation.
        """
        height, width = img_gray.shape[:2]
        upscaled = cv2.resize(
            img_gray, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC
        )
        return cv2.medianBlur(upscaled, 3)

    @staticmethod
    def denoise_colored(image: np.ndarray) -> np.ndarray:
        """
        Advanced denoising for colored images while preserving edges.
        """
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    @staticmethod
    def clean_for_ocr(img: np.ndarray) -> np.ndarray:
        """
        Advanced cleaning pipeline to specifically target digit clarity
        and remove pixel noise that causes misclassification.
        """
        gray = ImageEnhancer.to_gray(img)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Wrapper for ImageEnhancer.to_gray"""
    return ImageEnhancer.to_gray(img)


def sharpen(img_gray: np.ndarray) -> np.ndarray:
    """Wrapper for ImageEnhancer.sharpen"""
    return ImageEnhancer.sharpen(img_gray)


def denoise(img_gray: np.ndarray) -> np.ndarray:
    """Wrapper for ImageEnhancer.denoise"""
    return ImageEnhancer.denoise(img_gray)


def adaptive_threshold(img_gray: np.ndarray) -> np.ndarray:
    """Wrapper for ImageEnhancer.apply_threshold"""
    return ImageEnhancer.apply_threshold(img_gray)


def upscale_and_smooth(img_gray: np.ndarray, scale: int = 2) -> np.ndarray:
    """Wrapper for ImageEnhancer.upscale_and_smooth"""
    return ImageEnhancer.upscale_and_smooth(img_gray, scale)
