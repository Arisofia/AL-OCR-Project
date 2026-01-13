import pytesseract
import cv2
import numpy as np
import logging
from .enhance import ImageEnhancer
from .reconstruct import PixelReconstructor

logger = logging.getLogger("ocr-service.engine")

class IterativeOCREngine:
    """Manages the iterative OCR cycles and reconstruction feedback loop."""
    
    def __init__(self, iterations: int = 3):
        self.max_iterations = iterations
        self.enhancer = ImageEnhancer()
        self.reconstructor = PixelReconstructor()

    def process_image(self, image_bytes: bytes) -> dict:
        """Runs the iterative pipeline on the provided image bytes."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Invalid image format"}

        current_img = img.copy()
        best_text = ""
        iteration_history = []

        for i in range(self.max_iterations):
            logger.info(f"Starting iteration {i+1}/{self.max_iterations}")
            
            # 1. Enhancement Pass
            enhanced = self.enhancer.sharpen(current_img)
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            thresh = self.enhancer.apply_threshold(gray)
            
            # 2. OCR Attempt
            config = r'--oem 3 --psm 6 -l spa+eng'
            text = pytesseract.image_to_string(thresh, config=config).strip()
            
            # 3. Store Result
            iteration_history.append({
                "iteration": i + 1,
                "text_length": len(text),
                "preview_text": text[:50] + "..." if len(text) > 50 else text
            })

            # 4. Reconstruction Pass (Feedback)
            # In a real implementation, we would use the detected text positions 
            # to guide specific reconstruction on obscured areas.
            if len(text) > len(best_text):
                best_text = text
                
            # Prepare image for next iteration (progressive cleaning)
            current_img = cv2.detailEnhance(current_img, sigma_s=10, sigma_r=0.15)
            
        return {
            "text": best_text,
            "iterations": iteration_history,
            "success": len(best_text) > 0
        }
