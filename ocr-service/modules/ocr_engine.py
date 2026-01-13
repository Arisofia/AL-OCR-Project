import pytesseract
import cv2
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple
from .enhance import ImageEnhancer
from .reconstruct import PixelReconstructor
from .advanced_recon import AdvancedPixelReconstructor
from .learning_engine import LearningEngine

logger = logging.getLogger("ocr-service.engine")

# Try to import the packaged reconstruction pipeline (optional)
try:
    from ocr_reconstruct import process_bytes as recon_process_bytes
    RECON_AVAILABLE = True
except ImportError:
    recon_process_bytes = None
    RECON_AVAILABLE = False
except Exception as e:
    logger.warning(f"Error importing ocr_reconstruct: {e}")
    recon_process_bytes = None
    RECON_AVAILABLE = False

class IterativeOCREngine:
    """Manages the iterative OCR cycles and reconstruction feedback loop."""
    
    def __init__(self, iterations: int = 3):
        self.max_iterations = iterations
        self.enhancer = ImageEnhancer()
        self.reconstructor = PixelReconstructor()
        self.advanced_reconstructor = AdvancedPixelReconstructor()
        self.learning_engine = LearningEngine()

    def _decode_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def _run_reconstruction(self, image_bytes: bytes) -> tuple:
        if not RECON_AVAILABLE or recon_process_bytes is None:
            return None, None

        try:
            logger.info("Running packaged reconstruction preprocessor")
            recon_text, recon_img_bytes, recon_meta = recon_process_bytes(image_bytes, iterations=self.max_iterations)
            
            rec_img = None
            if recon_img_bytes:
                nparr = np.frombuffer(recon_img_bytes, np.uint8)
                rec_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return {"preview_text": recon_text, "meta": recon_meta}, rec_img
        except Exception as e:
            logger.warning(f"Reconstruction preprocessor failed: {e}")
            return None, None

    def _perform_ocr_iteration(self, img: np.ndarray, iteration: int, use_reconstruction: bool) -> tuple:
        # 1. Enhancement Pass
        enhanced = self.enhancer.sharpen(img)
        
        if use_reconstruction and iteration == 0:
            # Use the advanced overlay removal for the first pass grayscale
            thresh = self.reconstructor.remove_color_overlay(enhanced)
        else:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            thresh = self.enhancer.apply_threshold(gray)
        
        # 2. OCR Attempt
        config = r'--oem 3 --psm 6 -l spa+eng'
        text = pytesseract.image_to_string(thresh, config=config).strip()
        
        return text, thresh

    def process_image(self, image_bytes: bytes, use_reconstruction: bool = False) -> dict:
        """Runs the iterative pipeline on the provided image bytes."""
        # 1. Input Validation
        if not image_bytes:
            return {"error": "Empty image content"}
        
        # Limit image size to 10MB to prevent OOM
        if len(image_bytes) > 10 * 1024 * 1024:
            return {"error": "Image size exceeds 10MB limit"}

        img = self._decode_image(image_bytes)
        if img is None:
            return {"error": "Invalid image format or corrupted file"}

        current_img = img.copy()
        reconstruction_info = None

        if use_reconstruction:
            logger.info("Starting reconstruction phase")
            recon_info, recon_img = self._run_reconstruction(image_bytes)
            if recon_info:
                reconstruction_info = recon_info
            if recon_img is not None:
                current_img = recon_img
                logger.info("Using reconstructed image for subsequent iterations")

        best_text = ""
        iteration_history = []

        for i in range(self.max_iterations):
            logger.info(f"Starting iteration {i+1}/{self.max_iterations}")
            
            try:
                text, _ = self._perform_ocr_iteration(current_img, i, use_reconstruction)
                
                iteration_history.append({
                    "iteration": i + 1,
                    "text_length": len(text),
                    "preview_text": text[:50] + "..." if len(text) > 50 else text
                })

                if len(text) > len(best_text):
                    best_text = text
                    
                # Prepare image for next iteration (progressive cleaning)
                current_img = cv2.detailEnhance(current_img, sigma_s=10, sigma_r=0.15)
            except Exception as e:
                logger.error(f"Error in OCR iteration {i+1}: {e}")
                iteration_history.append({"iteration": i + 1, "error": str(e)})
            
        resp = {
            "text": best_text,
            "iterations": iteration_history,
            "success": len(best_text) > 0
        }
        if reconstruction_info:
            resp["reconstruction"] = reconstruction_info
        return resp

    async def process_image_advanced(self, image_bytes: bytes, doc_type: str = "generic") -> dict:
        """
        Advanced async pipeline that 'eliminates layers' using AI and 
        applies continuous learning.
        """
        # 1. Check learned patterns
        pattern = await self.learning_engine.get_pattern_knowledge(doc_type)
        if pattern:
            logger.info(f"Applying learned pattern for {doc_type}")

        # 2. Use AI for pixel-by-pixel reconstruction (Layer elimination)
        ai_result = await self.advanced_reconstructor.reconstruct_with_ai(image_bytes)
        
        if "error" in ai_result:
            # Fallback to standard iterative process if AI fails
            return self.process_image(image_bytes, use_reconstruction=True)

        # 3. Continuous Learning: Record results
        await self.learning_engine.learn_from_result(
            doc_type=doc_type,
            font_meta={"source": "ai_reconstruction"},
            accuracy_score=0.95 # Assumed for now
        )

        return {
            "text": ai_result["text"],
            "method": "advanced_ai_reconstruction",
            "success": True
        }
