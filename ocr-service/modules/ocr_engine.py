"""
Core OCR engine managing iterative processing and reconstruction loops.

This module coordinates image enhancement, layout analysis, 
reconstruction, and OCR iterations to achieve high-confidence results.
"""

import logging
from typing import Optional, Any, Tuple, Dict, List

import cv2
import numpy as np
import pytesseract

try:
    from ocr_reconstruct.modules.enhance import ImageEnhancer
    from ocr_reconstruct.modules.reconstruct import PixelReconstructor
except ImportError:
    ImageEnhancer = None
    PixelReconstructor = None

from .advanced_recon import AdvancedPixelReconstructor
from .learning_engine import LearningEngine
from .layout import DocumentLayoutAnalyzer
from .confidence import ConfidenceScorer
from .ocr_config import TesseractConfig, EngineConfig
from .image_toolkit import ImageToolkit
from services.textract import TextractService

logger = logging.getLogger("ocr-service.engine")

# Try to import the packaged reconstruction pipeline (optional)
try:
    from ocr_reconstruct import process_bytes as recon_process_bytes
    RECON_AVAILABLE = True
except ImportError:
    recon_process_bytes = None
    RECON_AVAILABLE = False


class IterativeOCREngine:
    """
    Manages the iterative OCR cycles and reconstruction feedback loop.
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        enhancer: Optional[Any] = None,
        reconstructor: Optional[Any] = None,
        advanced_reconstructor: Optional[AdvancedPixelReconstructor] = None,
        learning_engine: Optional[LearningEngine] = None,
        textract_service: Optional[TextractService] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        ocr_config: Optional[TesseractConfig] = None
    ):
        """
        Initializes the OCR engine with necessary components and configuration.
        """
        self.config = config or EngineConfig()
        self.enhancer = enhancer or (ImageEnhancer() if ImageEnhancer else None)
        self.reconstructor = (
            reconstructor or (PixelReconstructor() if PixelReconstructor else None)
        )
        self.advanced_reconstructor = (
            advanced_reconstructor or AdvancedPixelReconstructor()
        )
        self.learning_engine = learning_engine or LearningEngine()
        self.textract_service = textract_service or TextractService()
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self.ocr_config = ocr_config or TesseractConfig()

    def _run_reconstruction(
        self,
        image_bytes: bytes
    ) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Runs the optional reconstruction preprocessor.
        """
        if not RECON_AVAILABLE or recon_process_bytes is None:
            return None, None

        try:
            logger.info("Running packaged reconstruction preprocessor")
            recon_text, recon_img_bytes, recon_meta = recon_process_bytes(
                image_bytes,
                iterations=self.config.max_iterations
            )

            rec_img = None
            if recon_img_bytes:
                rec_img = ImageToolkit.decode_image(recon_img_bytes)

            return {"preview_text": recon_text, "meta": recon_meta}, rec_img
        except Exception as e:
            logger.warning("Reconstruction preprocessor failed: %s", e)
            return None, None

    def _perform_ocr_iteration(
        self,
        img: np.ndarray,
        iteration: int,
        use_reconstruction: bool
    ) -> Tuple[str, np.ndarray]:
        """
        Executes a single OCR iteration including enhancement and thresholding.
        """
        # 1. Enhancement Pass
        enhanced = (
            self.enhancer.sharpen(img) if self.enhancer else img
        )

        if use_reconstruction and iteration == 0 and self.reconstructor:
            # Use advanced overlay removal for the first pass
            img = self.reconstructor.remove_redactions(enhanced)
            thresh = self.reconstructor.remove_color_overlay(img)
        else:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            thresh = (
                self.enhancer.apply_threshold(gray)
                if self.enhancer else gray
            )

        # 2. Whole-image OCR
        try:
            text = pytesseract.image_to_string(
                thresh, config=self.ocr_config.flags
            ).strip()
        except Exception as e:
            logger.error("Tesseract whole-page OCR failed: %s", e)
            text = ""

        return text, thresh

    def _ocr_regions(self, thresh: np.ndarray, regions: List[Dict[str, Any]]) -> str:
        """
        Performs OCR on specific document regions and combines the text.
        """
        combined_text = []

        for region in regions:
            x, y, w, h = region["bbox"]
            roi = thresh[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            # Add some padding to ROI for better Tesseract performance
            roi = ImageToolkit.prepare_roi(roi)

            try:
                text = pytesseract.image_to_string(
                    roi, config=self.ocr_config.flags
                ).strip()
                if text:
                    combined_text.append(text)
            except Exception as e:
                logger.warning("Tesseract region OCR failed for region %s: %s", region.get("id"), e)

        return "\n\n".join(combined_text)

    def process_image(self, image_bytes: bytes, 
                      use_reconstruction: bool = False) -> dict:
        """
        Runs the iterative pipeline on the provided image bytes.
        """
        # 1. Input Validation
        validation_error = ImageToolkit.validate_image(
            image_bytes, max_size_mb=self.config.max_image_size_mb
        )
        if validation_error:
            return {"error": validation_error}

        img = ImageToolkit.decode_image(image_bytes)
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
                logger.info("Using reconstructed image for iterations")

        best_text = ""
        best_confidence = 0.0
        iteration_history = []

        # Initial layout analysis
        layout_regions = DocumentLayoutAnalyzer.detect_regions(image_bytes)

        for i in range(self.config.max_iterations):
            logger.info("Starting iteration %s/%s", i + 1, 
                        self.config.max_iterations)

            try:
                text, thresh = self._perform_ocr_iteration(
                    current_img, i, use_reconstruction
                )

                # Region-based fallback for low confidence
                should_use_regions = (
                    i == 1 and 
                    best_confidence < self.config.confidence_threshold and 
                    len(layout_regions) > 1
                )
                
                if should_use_regions:
                    logger.info("Confidence low, attempting region-based OCR")
                    text = self._ocr_regions(thresh, layout_regions)

                confidence = self.confidence_scorer.calculate(text)

                preview = text[:50] + "..." if len(text) > 50 else text
                iteration_history.append({
                    "iteration": i + 1,
                    "text_length": len(text),
                    "confidence": confidence,
                    "method": "region-based" if should_use_regions else "full-page",
                    "preview_text": preview
                })

                if confidence > best_confidence:
                    best_text = text
                    best_confidence = confidence

                # Prepare image for next iteration
                current_img = ImageToolkit.enhance_iteration(current_img)
            except Exception as e:
                logger.error("Error in OCR iteration %s: %s", i + 1, e)
                iteration_history.append({"iteration": i + 1, "error": str(e)})

        resp = {
            "text": best_text,
            "confidence": best_confidence,
            "iterations": iteration_history,
            "success": len(best_text) > 0
        }
        if reconstruction_info:
            resp["reconstruction"] = reconstruction_info
        return resp

    async def process_image_advanced(
        self,
        image_bytes: bytes,
        doc_type: Optional[str] = None
    ) -> dict:
        """
        Advanced async pipeline that applies AI reconstruction and learning.
        """
        doc_type = doc_type or self.config.default_doc_type
        
        # 0. Validate input to avoid passing empty or oversized data to the AI reconstructor
        validation_error = ImageToolkit.validate_image(
            image_bytes, max_size_mb=self.config.max_image_size_mb
        )
        if validation_error:
            return {"error": validation_error}

        # 1. Analyze Layout
        layout_regions = DocumentLayoutAnalyzer.detect_regions(image_bytes)
        layout_type = DocumentLayoutAnalyzer.classify_layout(layout_regions)
        logger.info(
            "Detected layout type: %s with %s regions",
            layout_type,
            len(layout_regions)
        )

        # 2. Check learned patterns
        pattern = await self.learning_engine.get_pattern_knowledge(doc_type)
        if pattern:
            logger.info("Applying learned pattern for %s", doc_type)

        # 3. Layer elimination using AI reconstruction
        context = pattern or {}
        context.update({
            "layout_type": layout_type,
            "region_count": len(layout_regions)
        })

        ai_result = await self.advanced_reconstructor.reconstruct_with_ai(
            image_bytes,
            context=context
        )

        if "error" in ai_result:
            # Fallback to standard iterative process if AI fails
            return self.process_image(image_bytes, use_reconstruction=True)

        # 4. Defensive text extraction and confidence calculation
        extracted_text = ai_result.get("text", "")
        confidence = self.confidence_scorer.calculate(extracted_text)

        # 5. Record results for continuous learning
        await self.learning_engine.learn_from_result(
            doc_type=doc_type,
            font_meta={
                "source": "ai_reconstruction",
                "model": ai_result.get("model", "unknown"),
                "layout": layout_type
            },
            accuracy_score=confidence
        )

        return {
            "text": extracted_text,
            "method": "advanced_ai_reconstruction",
            "confidence": confidence,
            "layout_analysis": {
                "type": layout_type,
                "regions": len(layout_regions)
            },
            "success": True
        }
