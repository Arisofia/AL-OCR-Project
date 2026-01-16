"""
Core OCR Orchestration Engine for high-fidelity document intelligence.
Manages iterative cycles, layout analysis, and pixel reconstruction.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract  # type: ignore

from ocr_reconstruct import process_bytes as recon_process_bytes
from ocr_reconstruct.modules.enhance import ImageEnhancer
from ocr_reconstruct.modules.reconstruct import PixelReconstructor

from .advanced_recon import AdvancedPixelReconstructor
from .confidence import ConfidenceScorer
from .image_toolkit import ImageToolkit
from .layout import DocumentLayoutAnalyzer
from .learning_engine import LearningEngine
from .ocr_config import EngineConfig, TesseractConfig

RECON_AVAILABLE = True

logger = logging.getLogger("ocr-service.engine")


class IterativeOCREngine:
    """
    Advanced OCR Engine utilizing iterative feedback loops
    and layer elimination techniques.
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        enhancer: Optional[Any] = None,
        reconstructor: Optional[Any] = None,
        advanced_reconstructor: Optional[AdvancedPixelReconstructor] = None,
        learning_engine: Optional[LearningEngine] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        ocr_config: Optional[TesseractConfig] = None,
    ):
        """
        Dependency-injected initializer for the core orchestration engine.
        """
        self.config = config or EngineConfig()
        self.enhancer = enhancer or ImageEnhancer()
        self.reconstructor = reconstructor or (
            PixelReconstructor() if PixelReconstructor is not None else None
        )
        self.advanced_reconstructor = (
            advanced_reconstructor or AdvancedPixelReconstructor()
        )
        self.learning_engine = learning_engine or LearningEngine()
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
        except Exception as e:
            logger.warning("Reconstruction pipeline failed: %s", e)
            return None, None

    def _preprocess_for_ocr(
        self, img: np.ndarray, iteration: int, use_reconstruction: bool
    ) -> np.ndarray:
        """Encapsulates image enhancement and layer management logic."""
        enhanced = self.enhancer.sharpen(img) if self.enhancer else img

        if use_reconstruction and iteration == 0 and self.reconstructor:
            # Apply advanced overlay elimination for initial high-signal pass
            rectified = self.reconstructor.remove_redactions(enhanced)
            enhanced = self.reconstructor.remove_color_overlay(rectified)

        gray = (
            cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            if len(enhanced.shape) == 3
            else enhanced
        )
        return self.enhancer.apply_threshold(gray) if self.enhancer else gray

    def _perform_ocr_iteration(
        self, img: np.ndarray, iteration: int, use_reconstruction: bool
    ) -> Tuple[str, np.ndarray]:
        """
        Single-cycle OCR execution: Enhancement -> Thresholding -> Extraction.
        """
        thresh = self._preprocess_for_ocr(img, iteration, use_reconstruction)

        # Phase 2: Whole-document text extraction
        try:
            text = pytesseract.image_to_string(thresh, config=self.ocr_config.flags)
            text = text.strip()
        except (pytesseract.TesseractError, RuntimeError):
            logger.error("Extraction failed at iteration %s", iteration)
            text = ""
        except Exception:
            logger.exception("Unexpected error during iteration %s", iteration)
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
                text = pytesseract.image_to_string(
                    roi, config=self.ocr_config.flags
                ).strip()
                if text:
                    combined_text.append(text)
            except (pytesseract.TesseractError, RuntimeError):
                logger.warning(
                    "Targeted extraction failed | RegionId: %s",
                    region.get("id"),
                )
            except Exception:
                logger.exception("Critical failure in region-based extraction")

        return "\n\n".join(combined_text)

    def _get_ocr_input_image(
        self, image_bytes: bytes, use_reconstruction: bool
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Determines the best source image for OCR iterations."""
        img = ImageToolkit.decode_image(image_bytes)
        if img is None:
            return None, None

        reconstruction_info = None
        current_img = img.copy()

        if use_reconstruction:
            logger.info("Initiating pixel reconstruction sequence")
            recon_info, recon_img = self._run_reconstruction(image_bytes)
            if recon_info:
                reconstruction_info = recon_info
            if recon_img is not None:
                current_img = recon_img
                logger.info("Using high-fidelity reconstructed source")

        return current_img, reconstruction_info

    def _validate_image_input(self, image_bytes: bytes) -> Optional[Dict[str, str]]:
        """Common validation for OCR input."""
        validation_error = ImageToolkit.validate_image(
            image_bytes, max_size_mb=self.config.max_image_size_mb
        )
        return {"error": validation_error} if validation_error else {}

    def _should_use_region_ocr(
        self, iteration: int, confidence: float, region_count: int
    ) -> bool:
        """Determines if adaptive region-based extraction should be triggered."""
        return (
            iteration == 1
            and confidence < self.config.confidence_threshold
            and region_count > 1
        )

    def process_image(
        self,
        image_bytes: bytes,
        use_reconstruction: bool = False,
    ) -> dict:
        """
        Executes iterative OCR pipeline with confidence-based feedback.
        """
        if error_resp := self._validate_image_input(image_bytes):
            return error_resp

        current_img, recon_info = self._get_ocr_input_image(
            image_bytes, use_reconstruction
        )
        if current_img is None:
            return {"error": "Corrupted or unsupported image format"}

        best_text, best_confidence = "", 0.0
        iteration_history = []
        layout_regions = DocumentLayoutAnalyzer.detect_regions(image_bytes)

        for i in range(self.config.max_iterations):
            text, confidence, method, current_img = self._run_single_iteration(
                current_img, i, use_reconstruction, best_confidence, layout_regions
            )

            iteration_history.append(
                {
                    "iteration": i + 1,
                    "text_length": len(text),
                    "confidence": confidence,
                    "method": method,
                    "preview_text": f"{text[:50]}..." if len(text) > 50 else text,
                }
                if text is not None
                else {"iteration": i + 1, "error": "failed"}
            )

            if text and confidence > best_confidence:
                best_text, best_confidence = text, confidence

        return self._build_process_response(
            best_text, best_confidence, iteration_history, recon_info
        )

    def _run_single_iteration(
        self,
        current_img: np.ndarray,
        iteration: int,
        use_reconstruction: bool,
        best_confidence: float,
        layout_regions: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], float, str, np.ndarray]:
        """Runs a single OCR iteration with adaptive strategies."""
        logger.info(
            "Iteration loop | Progress: %s/%s",
            iteration + 1,
            self.config.max_iterations,
        )
        try:
            text, thresh = self._perform_ocr_iteration(
                current_img, iteration, use_reconstruction
            )

            # Adaptive Strategy: Fallback to region-based if confidence is low
            is_region_pass = self._should_use_region_ocr(
                iteration, best_confidence, len(layout_regions)
            )
            if is_region_pass:
                logger.info("Engaging targeted region extraction")
                text = self._ocr_regions(thresh, layout_regions)

            confidence = self.confidence_scorer.calculate(text)
            method = "region-based" if is_region_pass else "full-page"
            next_img = ImageToolkit.enhance_iteration(current_img)

            return text, confidence, method, next_img
        except Exception:
            logger.error("Failure in iteration %s", iteration + 1)
            return None, 0.0, "failed", current_img

    def _build_process_response(
        self,
        best_text: str,
        best_confidence: float,
        iteration_history: List[Dict[str, Any]],
        recon_info: Optional[Dict[str, Any]],
    ) -> dict:
        """Constructs the final response dictionary."""
        resp = {
            "text": best_text,
            "confidence": best_confidence,
            "iterations": iteration_history,
            "success": len(best_text) > 0,
        }
        if recon_info:
            resp["reconstruction"] = recon_info
        return resp

    async def process_image_advanced(
        self, image_bytes: bytes, doc_type: Optional[str] = None
    ) -> dict:
        """
        AI pipeline with reconstruction and continuous learning.
        """
        if error_resp := self._validate_image_input(image_bytes):
            return error_resp

        doc_type = doc_type or self.config.default_doc_type

        # Step 1: Structural Layout Analysis
        def _analyze_layout():
            regions = DocumentLayoutAnalyzer.detect_regions(image_bytes)
            l_type = DocumentLayoutAnalyzer.classify_layout(regions)
            return regions, l_type

        layout_regions, layout_type = await asyncio.to_thread(_analyze_layout)
        logger.info("Structural analysis complete | Regions: %s", len(layout_regions))

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
            return await asyncio.to_thread(
                self.process_image, image_bytes, use_reconstruction=True
            )

        # Step 4: Verification and Intelligence Aggregation
        extracted_text = ai_result.get("text", "")
        confidence = self.confidence_scorer.calculate(extracted_text)

        # Step 5: Autonomous Feedback Loop
        asyncio.create_task(
            self.learning_engine.learn_from_result(
                doc_type=doc_type,
                font_meta={
                    "source": "ai_reconstruction",
                    "model": ai_result.get("model", "unknown"),
                    "layout": layout_type,
                },
                accuracy_score=confidence,
            )
        )

        return {
            "text": extracted_text,
            "method": "advanced_ai_reconstruction",
            "confidence": confidence,
            "layout_analysis": {"type": layout_type, "regions": len(layout_regions)},
            "success": True,
        }
