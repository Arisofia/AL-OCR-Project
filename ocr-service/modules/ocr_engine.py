"""
Core OCR Orchestration Engine for high-fidelity document intelligence.
Manages the iterative extraction cycle, layout analysis, and AI-driven pixel reconstruction.
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

# Runtime check for optional reconstruction pipeline
try:
    from ocr_reconstruct import process_bytes as recon_process_bytes
    RECON_AVAILABLE = True
except ImportError:
    recon_process_bytes = None
    RECON_AVAILABLE = False


class IterativeOCREngine:
    """
    Advanced OCR Engine utilizing iterative feedback loops and layer elimination techniques.
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
        ocr_config: Optional[TesseractConfig] = None,
    ):
        """
        Dependency-injected initializer for the core orchestration engine.
        """
        self.config = config or EngineConfig()
        self.enhancer = enhancer or (ImageEnhancer() if ImageEnhancer else None)
        self.reconstructor = reconstructor or (
            PixelReconstructor() if PixelReconstructor else None
        )
        self.advanced_reconstructor = (
            advanced_reconstructor or AdvancedPixelReconstructor()
        )
        self.learning_engine = learning_engine or LearningEngine()
        self.textract_service = textract_service or TextractService()
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self.ocr_config = ocr_config or TesseractConfig()

    def _run_reconstruction(
        self, image_bytes: bytes
    ) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Executes the pixel reconstruction preprocessor to eliminate visual noise.
        """
        if not RECON_AVAILABLE or recon_process_bytes is None:
            return None, None

        try:
            logger.info("Executing reconstruction preprocessor pipeline")
            recon_text, recon_img_bytes, recon_meta = recon_process_bytes(
                image_bytes, iterations=self.config.max_iterations
            )

            rec_img = None
            if recon_img_bytes:
                rec_img = ImageToolkit.decode_image(recon_img_bytes)

            return {"preview_text": recon_text, "meta": recon_meta}, rec_img
        except (ValueError, TypeError, RuntimeError) as e:
            logger.warning("Reconstruction pipeline operational failure: %s", e)
            return None, None
        except Exception:
            logger.exception("Unexpected system failure in reconstruction pipeline")
            return None, None

    def _perform_ocr_iteration(
        self, img: np.ndarray, iteration: int, use_reconstruction: bool
    ) -> Tuple[str, np.ndarray]:
        """
        Single-cycle OCR execution: Enhancement -> Thresholding -> Extraction.
        """
        # Phase 1: Enhancement and Layer Management
        enhanced = (
            self.enhancer.sharpen(img) if self.enhancer else img
        )

        if use_reconstruction and iteration == 0 and self.reconstructor:
            # Apply advanced overlay elimination for initial high-signal pass
            img = self.reconstructor.remove_redactions(enhanced)
            thresh = self.reconstructor.remove_color_overlay(img)
        else:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            thresh = self.enhancer.apply_threshold(gray) if self.enhancer else gray

        # Phase 2: Whole-document text extraction
        try:
            text = pytesseract.image_to_string(thresh, config=self.ocr_config.flags)
            text = text.strip()
        except pytesseract.TesseractError as e:
            logger.error("Tesseract failure at iteration %s: %s", iteration, e)
            text = ""
        except Exception:
            logger.exception("Unexpected extraction failure at iteration %s", iteration)
            text = ""

        return text, thresh

    def _ocr_regions(self, thresh: np.ndarray, regions: List[Dict[str, Any]]) -> str:
        """
        Region-of-Interest (ROI) targeted extraction for complex document layouts.
        """
        combined_text = []

        for region in regions:
            x, y, w, h = region["bbox"]
            roi = thresh[y : y + h, x : x + w]
            if roi.size == 0:
                continue

            # Standardize ROI padding for optimal Tesseract alignment
            roi = ImageToolkit.prepare_roi(roi)

            try:
                text = pytesseract.image_to_string(roi, config=self.ocr_config.flags)
                text = text.strip()
                if text:
                    combined_text.append(text)
            except pytesseract.TesseractError as e:
                logger.warning(
                    "Region extraction failure | RegionId: %s | Error: %s",
                    region.get("id"),
                    e,
                )
            except Exception:
                logger.exception("Unexpected region extraction failure | RegionId: %s", region.get("id"))

        return "\n\n".join(combined_text)

    def process_image(
        self,
        image_bytes: bytes,
        use_reconstruction: bool = False,
    ) -> dict:
        """
        Executes the iterative OCR pipeline with confidence-based feedback loops.
        """
        # Stage 1: Payload Validation and Pre-processing
        validation_error = ImageToolkit.validate_image(
            image_bytes, max_size_mb=self.config.max_image_size_mb
        )
        if validation_error:
            return {"error": validation_error}

        img = ImageToolkit.decode_image(image_bytes)
        if img is None:
            return {"error": "Corrupted or unsupported image format"}

        current_img = img.copy()
        reconstruction_info = None

        # Stage 2: Pixel Reconstruction (Optional high-fidelity pass)
        if use_reconstruction:
            logger.info("Initiating pixel reconstruction sequence")
            recon_info, recon_img = self._run_reconstruction(image_bytes)
            if recon_info:
                reconstruction_info = recon_info
            if recon_img is not None:
                current_img = recon_img
                logger.info("Using high-fidelity reconstructed source for iterations")

        best_text = ""
        best_confidence = 0.0
        iteration_history = []

        # Stage 3: Document Layout Decomposition
        layout_regions = DocumentLayoutAnalyzer.detect_regions(image_bytes)

        # Stage 4: Iterative Optimization Loop
        for i in range(self.config.max_iterations):
            logger.info("Iteration loop | Progress: %s/%s", i + 1, self.config.max_iterations)

            try:
                text, thresh = self._perform_ocr_iteration(
                    current_img, i, use_reconstruction
                )

                # Execute fallback to targeted extraction if global confidence is insufficient
                should_use_regions = (
                    i == 1
                    and best_confidence < self.config.confidence_threshold
                    and len(layout_regions) > 1
                )

                if should_use_regions:
                    logger.info("Confidence below threshold | Engaging targeted region extraction")
                    text = self._ocr_regions(thresh, layout_regions)

                confidence = self.confidence_scorer.calculate(text)

                preview = text[:50] + "..." if len(text) > 50 else text
                method = "region-based" if should_use_regions else "full-page"
                
                iteration_history.append({
                    "iteration": i + 1,
                    "text_length": len(text),
                    "confidence": confidence,
                    "method": method,
                    "preview_text": preview
                })

                if confidence > best_confidence:
                    best_text = text
                    best_confidence = confidence

                # Apply iterative enhancement for subsequent cycle
                current_img = ImageToolkit.enhance_iteration(current_img)
            except (ValueError, TypeError, RuntimeError, pytesseract.TesseractError) as e:
                logger.error("Operational failure in iteration %s: %s", i + 1, e)
                iteration_history.append({"iteration": i + 1, "error": str(e)})
            except Exception:
                logger.exception("Unexpected system failure in iteration %d", i + 1)
                iteration_history.append({"iteration": i + 1, "error": "unexpected_system_failure"})

        resp = {
            "text": best_text,
            "confidence": best_confidence,
            "iterations": iteration_history,
            "success": len(best_text) > 0,
        }
        if reconstruction_info:
            resp["reconstruction"] = reconstruction_info
        return resp

    async def process_image_advanced(
        self, image_bytes: bytes, doc_type: Optional[str] = None
    ) -> dict:
        """
        Advanced autonomous pipeline leveraging AI reconstruction and continuous learning.
        """
        doc_type = doc_type or self.config.default_doc_type

        validation_error = ImageToolkit.validate_image(
            image_bytes, max_size_mb=self.config.max_image_size_mb
        )
        if validation_error:
            return {"error": validation_error}

        # Step 1: Structural Layout Analysis
        layout_regions = DocumentLayoutAnalyzer.detect_regions(image_bytes)
        layout_type = DocumentLayoutAnalyzer.classify_layout(layout_regions)
        logger.info("Structural analysis complete | Type: %s | Regions: %s", layout_type, len(layout_regions))

        # Step 2: Contextual Knowledge Retrieval
        pattern = await self.learning_engine.get_pattern_knowledge(doc_type)
        if pattern:
            logger.info("Applying domain-specific patterns for %s", doc_type)

        # Step 3: AI-Driven Layer Elimination
        context = pattern or {}
        context.update(
            {"layout_type": layout_type, "region_count": len(layout_regions)}
        )

        ai_result = await self.advanced_reconstructor.reconstruct_with_ai(
            image_bytes, context=context
        )

        if "error" in ai_result:
            logger.warning("AI reconstruction failed | Triggering iterative fallback")
            return self.process_image(image_bytes, use_reconstruction=True)

        # Step 4: Verification and Intelligence Aggregation
        extracted_text = ai_result.get("text", "")
        confidence = self.confidence_scorer.calculate(extracted_text)

        # Step 5: Autonomous Feedback Loop
        await self.learning_engine.learn_from_result(
            doc_type=doc_type,
            font_meta={
                "source": "ai_reconstruction",
                "model": ai_result.get("model", "unknown"),
                "layout": layout_type,
            },
            accuracy_score=confidence,
        )

        return {
            "text": extracted_text,
            "method": "advanced_ai_reconstruction",
            "confidence": confidence,
            "layout_analysis": {"type": layout_type, "regions": len(layout_regions)},
            "success": True,
        }
