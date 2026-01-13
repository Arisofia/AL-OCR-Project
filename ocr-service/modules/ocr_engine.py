import pytesseract
import cv2
import numpy as np
import logging
from .enhance import ImageEnhancer
from .reconstruct import PixelReconstructor

logger = logging.getLogger("ocr-service.engine")

# Try to import the packaged reconstruction pipeline (optional)
try:
    from ocr_reconstruct.modules.pipeline import process_bytes as recon_process_bytes  # type: ignore
    RECON_AVAILABLE = True
except Exception:
    recon_process_bytes = None
    RECON_AVAILABLE = False

class IterativeOCREngine:
    """Manages the iterative OCR cycles and reconstruction feedback loop."""
    
    def __init__(self, iterations: int = 3):
        self.max_iterations = iterations
        self.enhancer = ImageEnhancer()
        self.reconstructor = PixelReconstructor()

    def process_image(self, image_bytes: bytes, use_reconstruction: bool = False) -> dict:
        """Runs the iterative pipeline on the provided image bytes.

        If reconstruction is requested and the optional package is available, use it
        to pre-process the image bytes (in-memory) and include reconstruction
        metadata in the returned dict as `reconstruction`.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Invalid image format"}

        current_img = img.copy()
        reconstruction_info = None

        # Optional reconstruction preprocessor using the packaged pipeline
        if use_reconstruction and RECON_AVAILABLE and recon_process_bytes is not None:
            try:
                logger.info("Running packaged reconstruction preprocessor")
                recon_text, recon_img_bytes, recon_meta = recon_process_bytes(image_bytes, iterations=self.max_iterations)
                reconstruction_info = {"preview_text": recon_text, "meta": recon_meta}
                if recon_img_bytes:
                    # Replace current image with reconstructed image
                    nparr2 = np.frombuffer(recon_img_bytes, np.uint8)
                    rec_img = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
                    if rec_img is not None:
                        current_img = rec_img
            except Exception as e:
                logger.warning(f"Reconstruction preprocessor failed: {e}")

        best_text = ""
        iteration_history = []

        for i in range(self.max_iterations):
            logger.info(f"Starting iteration {i+1}/{self.max_iterations}")
            
            # 1. Enhancement Pass
            enhanced = self.enhancer.sharpen(current_img)
            
            if use_reconstruction and i == 0:
                # Use the advanced overlay removal for the first pass grayscale
                thresh = self.reconstructor.remove_color_overlay(enhanced)
            else:
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
            
        resp = {
            "text": best_text,
            "iterations": iteration_history,
            "success": len(best_text) > 0
        }
        if reconstruction_info:
            resp["reconstruction"] = reconstruction_info
        return resp
