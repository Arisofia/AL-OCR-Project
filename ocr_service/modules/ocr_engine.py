"""
Core OCR Orchestration Engine for high-fidelity document intelligence.
Refactored into modular components for better maintainability and performance.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import cv2
import httpx
import numpy as np
import pytesseract  # type: ignore

from ocr_reconstruct import process_bytes as recon_process_bytes
from ocr_reconstruct.modules.enhance import ImageEnhancer
from ocr_reconstruct.modules.reconstruct import PixelReconstructor
from ocr_service.utils.capabilities import CapabilityProvider

from .advanced_recon import AdvancedPixelReconstructor
from .confidence import ConfidenceScorer
from .image_toolkit import ImageToolkit
from .layout import DocumentLayoutAnalyzer
from .learning_engine import LearningEngine
from .ocr_config import EngineConfig, TesseractConfig

__all__ = ["DocumentContext", "DocumentProcessor", "IterativeOCREngine"]

logger = logging.getLogger("ocr-service.engine")


@dataclass
class DocumentContext:
    """
    Holds state and intermediate results for a single document
    processing lifecycle.
    """

    image_bytes: bytes
    use_reconstruction: bool
    doc_type: str = "generic"
    original_img: Optional[np.ndarray] = None
    current_img: Optional[np.ndarray] = None
    layout_regions: list[dict[str, Any]] = field(default_factory=list)
    layout_type: str = "unknown"
    reconstruction_info: Optional[dict[str, Any]] = None
    best_text: str = ""
    best_confidence: float = 0.0
    iteration_history: list[dict[str, Any]] = field(default_factory=list)


class DocumentProcessor:
    """Handles low-level image processing and OCR extraction tasks."""

    def __init__(
        self,
        enhancer: ImageEnhancer,
        ocr_config: TesseractConfig,
        reconstructor: Optional[PixelReconstructor] = None,
    ):
        self.enhancer = enhancer
        self.ocr_config = ocr_config
        self.reconstructor = reconstructor

    async def decode_and_validate(self, ctx: DocumentContext) -> bool:
        """Decodes image and performs initial validation."""
        ctx.original_img = await ImageToolkit.decode_image_async(ctx.image_bytes)
        if ctx.original_img is None:
            return False
        ctx.current_img = ctx.original_img.copy()
        return True

    async def run_reconstruction(self, ctx: DocumentContext, max_iterations: int):
        """Executes reconstruction preprocessor if enabled."""
        if (
            not ctx.use_reconstruction
            or not CapabilityProvider.is_reconstruction_available()
            or recon_process_bytes is None
        ):
            return

        try:
            logger.info("Executing reconstruction preprocessor pipeline")
            # process_bytes is likely CPU bound, run in thread
            recon_text, recon_img_bytes, recon_meta = await asyncio.to_thread(
                recon_process_bytes, ctx.image_bytes, iterations=max_iterations
            )

            if recon_img_bytes:
                ctx.current_img = await ImageToolkit.decode_image_async(recon_img_bytes)
                ctx.reconstruction_info = {
                    "preview_text": recon_text,
                    "meta": recon_meta,
                }
                logger.info("Using high-fidelity reconstructed source")
        except Exception as e:
            logger.warning("Reconstruction pipeline failed: %s", e)

    def preprocess_frame(
        self, img: np.ndarray, iteration: int, use_recon: bool
    ) -> np.ndarray:
        """Applies sharpening, layer elimination, and thresholding."""
        enhanced = self.enhancer.sharpen(img)

        if use_recon and iteration == 0 and self.reconstructor:
            rectified = self.reconstructor.remove_redactions(enhanced)
            enhanced = self.reconstructor.remove_color_overlay(rectified)

        if len(enhanced.shape) == 3:
            gray = cast(np.ndarray, cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
        else:
            gray = enhanced
        return self.enhancer.apply_threshold(gray)

    async def extract_text(
        self, img: np.ndarray, regions: Optional[list[dict[str, Any]]] = None
    ) -> str:
        """Performs OCR on the whole image or specific regions."""
        if regions:
            return await self._extract_from_regions(img, regions)

        try:
            return await asyncio.to_thread(
                pytesseract.image_to_string, img, config=self.ocr_config.flags
            )
        except Exception as e:
            logger.error("OCR whole image extraction failed: %s", e)
            return ""

    async def _extract_from_regions(
        self, img: np.ndarray, regions: list[dict[str, Any]]
    ) -> str:
        """Performs targeted extraction on ROIs."""
        combined_text = []
        for region in regions:
            x, y, w, h = region["bbox"]
            roi = img[y : y + h, x : x + w]
            if roi.size == 0:
                continue

            roi = ImageToolkit.prepare_roi(roi)
            try:
                text = await asyncio.to_thread(
                    pytesseract.image_to_string, roi, config=self.ocr_config.flags
                )
                if text.strip():
                    combined_text.append(text.strip())
            except Exception as e:
                logger.warning("Region extraction failed: %s", e)

        return "\n\n".join(combined_text)


class IterativeOCREngine:
    """
    Advanced OCR Engine utilizing iterative feedback loops.
    Acts as a Facade for DocumentProcessor and OCRPipeline logic.
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
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.config = config or EngineConfig()
        self._client = client
        self.processor = DocumentProcessor(
            enhancer=enhancer or ImageEnhancer(),
            ocr_config=ocr_config or TesseractConfig(),
            reconstructor=reconstructor
            or (
                PixelReconstructor()
                if CapabilityProvider.is_reconstruction_available()
                else None
            ),
        )
        self.advanced_reconstructor = advanced_reconstructor or AdvancedPixelReconstructor(
            client=self._client
        )
        self.learning_engine = learning_engine or LearningEngine()
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self._background_tasks: set[asyncio.Task] = set()

    async def close(self) -> None:
        """Cleanup engine resources."""
        await self.advanced_reconstructor.close()

    async def process_image(
        self, image_bytes: bytes, use_reconstruction: bool = False
    ) -> dict[str, Any]:
        """Entry point for standard iterative OCR pipeline."""
        validation_error = ImageToolkit.validate_image(
            image_bytes, self.config.max_image_size_mb
        )
        if validation_error:
            return {"error": validation_error}

        ctx = DocumentContext(
            image_bytes=image_bytes, use_reconstruction=use_reconstruction
        )

        if not await self.processor.decode_and_validate(ctx):
            return {"error": "Corrupted or unsupported image format"}

        # Initial Setup
        await asyncio.gather(
            self.processor.run_reconstruction(ctx, self.config.max_iterations),
            self._analyze_layout(ctx),
        )

        # Iteration Loop
        for i in range(self.config.max_iterations):
            await self._run_iteration(ctx, i)

        return self._build_response(ctx)

    async def process_image_advanced(
        self, image_bytes: bytes, doc_type: Optional[str] = None
    ) -> dict[str, Any]:
        """AI-driven pipeline with contextual learning."""
        validation_error = ImageToolkit.validate_image(
            image_bytes, self.config.max_image_size_mb
        )
        if validation_error:
            return {"error": validation_error}

        ctx = DocumentContext(
            image_bytes=image_bytes,
            use_reconstruction=True,
            doc_type=doc_type or self.config.default_doc_type,
        )

        # Parallel initialization: Layout + Learning Retrieval
        layout_task = self._analyze_layout(ctx)
        learning_task = self.learning_engine.get_pattern_knowledge(ctx.doc_type)

        await layout_task
        pattern = await learning_task

        # AI Reconstruction Pass
        context = (pattern or {}) | {
            "layout_type": ctx.layout_type,
            "region_count": len(ctx.layout_regions),
        }

        ai_result = await self.advanced_reconstructor.reconstruct_with_ai(
            image_bytes, context=context
        )

        if "error" in ai_result:
            logger.warning("AI reconstruction failed | Triggering iterative fallback")
            return await self.process_image(image_bytes, use_reconstruction=True)

        extracted_text = ai_result.get("text", "")
        confidence = self.confidence_scorer.calculate(extracted_text)

        # Learning Update (Background)
        self._schedule_learning(
            ctx.doc_type,
            ai_result.get("model", "unknown"),
            ctx.layout_type,
            confidence,
        )

        return {
            "text": extracted_text,
            "method": "advanced_ai_reconstruction",
            "confidence": confidence,
            "layout_analysis": {
                "type": ctx.layout_type,
                "regions": len(ctx.layout_regions),
            },
            "success": True,
        }

    async def _analyze_layout(self, ctx: DocumentContext):
        """Analyzes document layout in a thread."""

        def _run():
            regions = DocumentLayoutAnalyzer.detect_regions(ctx.image_bytes)
            l_type = DocumentLayoutAnalyzer.classify_layout(regions)
            return regions, l_type

        ctx.layout_regions, ctx.layout_type = await asyncio.to_thread(_run)

    async def _run_iteration(self, ctx: DocumentContext, i: int):
        """Executes a single iteration loop."""
        try:
            # Preprocess and Extract
            if ctx.current_img is None:
                raise ValueError("Context image is missing")

            thresh = self.processor.preprocess_frame(
                ctx.current_img, i, ctx.use_reconstruction
            )

            # Adaptive Strategy
            use_regions = (
                i == 1
                and ctx.best_confidence < self.config.confidence_threshold
                and len(ctx.layout_regions) > 1
            )
            text = await self.processor.extract_text(
                thresh, ctx.layout_regions if use_regions else None
            )

            # Score and Record
            confidence = self.confidence_scorer.calculate(text)
            ctx.iteration_history.append(
                {
                    "iteration": i + 1,
                    "text_length": len(text),
                    "confidence": confidence,
                    "method": "region-based" if use_regions else "full-page",
                    "preview_text": f"{text[:50]}..." if len(text) > 50 else text,
                }
            )

            if confidence > ctx.best_confidence:
                ctx.best_text, ctx.best_confidence = text, confidence

            # Prepare for next pass
            ctx.current_img = await ImageToolkit.enhance_iteration(ctx.current_img)
        except Exception as e:
            logger.error("Iteration %d failed: %s", i + 1, e)
            ctx.iteration_history.append({"iteration": i + 1, "error": "failed"})

    def _build_response(self, ctx: DocumentContext) -> dict[str, Any]:
        """Formats the final engine output."""
        resp = {
            "text": ctx.best_text,
            "confidence": ctx.best_confidence,
            "iterations": ctx.iteration_history,
            "success": len(ctx.best_text) > 0,
        }
        if ctx.reconstruction_info:
            resp["reconstruction"] = ctx.reconstruction_info
        return resp

    def _schedule_learning(self, doc_type: str, model: str, layout: str, score: float):
        """Schedules background task for learning."""
        task = asyncio.create_task(
            self.learning_engine.learn_from_result(
                doc_type=doc_type,
                font_meta={
                    "source": "ai_reconstruction",
                    "model": model,
                    "layout": layout,
                },
                accuracy_score=score,
            )
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
