import cv2
import numpy as np

class ImageEnhancer:
    """Handles image preprocessing for OCR enhancement."""
    
    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        """Applies a Laplacian sharpening kernel to enhance edges."""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def reduce_noise(image: np.ndarray) -> np.ndarray:
        """Applies median blur to reduce salt-and-pepper noise."""
        return cv2.medianBlur(image, 3)

    @staticmethod
    def apply_threshold(image: np.ndarray) -> np.ndarray:
        """Applies Otsu's binarization for high-contrast text isolation."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def denoise_colored(image: np.ndarray) -> np.ndarray:
        """Advanced denoising for colored images while preserving edges."""
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
