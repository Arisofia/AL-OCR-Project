"""
Core OCR Orchestration Engine for high-fidelity document intelligence.
Refactored into modular components for better maintainability and performance.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Optional, cast

import boto3
import cv2
import httpx
import numpy as np
import pytesseract  # type: ignore
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image, UnidentifiedImageError

from ocr_reconstruct import process_bytes as recon_process_bytes
from ocr_reconstruct.modules.enhance import ImageEnhancer
from ocr_reconstruct.modules.reconstruct import PixelReconstructor
from ocr_service.metrics import (
    OCR_ENGINE_PROCESS_IMAGE_ADVANCED_LATENCY,
    OCR_ENGINE_PROCESS_IMAGE_LATENCY,
    OCR_ERROR_COUNT,
    OCR_EXTRACTION_LATENCY,
    OCR_ITERATION_COUNT,
    OCR_RECONSTRUCTION_LATENCY,
)
from ocr_service.utils.capabilities import CapabilityProvider

from .advanced_recon import AdvancedPixelReconstructor
from .confidence import ConfidenceScorer
from .image_toolkit import ImageToolkit, ImageToolkitError
from .layout import DocumentLayoutAnalyzer, LayoutAnalysisError
from .learning_engine import LearningEngine
from .ocr_config import EngineConfig, TesseractConfig

__all__ = ["DocumentContext", "DocumentProcessor", "IterativeOCREngine"]

logger = logging.getLogger("ocr-service.engine")


def _configure_tesseract_cmd() -> Optional[str]:
    """Resolve a runnable tesseract binary path for Lambda/container runtimes."""
    candidates = [
        os.getenv("TESSERACT_CMD"),
        "/usr/bin/tesseract",
        "/opt/bin/tesseract",
        "/bin/tesseract",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate
    return None


_TESSERACT_CMD = _configure_tesseract_cmd()


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
        engine_config: EngineConfig,
        reconstructor: Optional[PixelReconstructor] = None,
    ):
        self.enhancer = enhancer
        self.ocr_config = ocr_config
        self.engine_config = engine_config
        self.reconstructor = reconstructor

    async def decode_and_validate(self, ctx: DocumentContext) -> bool:
        """Decodes image and performs initial validation."""
        try:
            ctx.original_img = await ImageToolkit.decode_image_async(ctx.image_bytes)
            if ctx.original_img is None:
                return False

            img = ctx.original_img

            # Adaptive upscaling before any further enhancement.
            if self.engine_config.max_upscale_factor > 1.0:
                img = ImageToolkit.upscale_for_ocr(
                    img,
                    max_upscale_factor=self.engine_config.max_upscale_factor,
                    max_long_side_px=self.engine_config.max_long_side_px,
                )

            ctx.current_img = img
            return True
        except Exception as e:
            logger.error("Initial image decode/preprocess failed: %s", e)
            OCR_ERROR_COUNT.labels(
                phase="decode_and_validate", error_type=type(e).__name__
            ).inc()
            return False

    async def extract_text_direct(self, image_bytes: bytes) -> str:
        """
        OpenCV-independent fallback OCR path using Pillow + Tesseract.

        This path is used when decoding/preprocessing fails in runtime
        environments with OpenCV binary incompatibilities.
        """
        try:
            with Image.open(BytesIO(image_bytes)) as pil_img:
                normalized = pil_img.convert("RGB")
                return await asyncio.to_thread(
                    pytesseract.image_to_string,
                    normalized,
                    config=self.ocr_config.flags,
                )
        except pytesseract.pytesseract.TesseractNotFoundError as e:
            logger.error(
                "Direct fallback OCR cannot run: tesseract binary unavailable "
                "(configured=%s): %s",
                _TESSERACT_CMD or "auto",
                e,
            )
            OCR_ERROR_COUNT.labels(
                phase="direct_fallback_tesseract", error_type=type(e).__name__
            ).inc()
            textract_text = await self.extract_text_textract(image_bytes)
            if textract_text:
                return textract_text
            return ""
        except UnidentifiedImageError as e:
            logger.error("Direct fallback could not open image bytes: %s", e)
            OCR_ERROR_COUNT.labels(
                phase="direct_fallback_open", error_type=type(e).__name__
            ).inc()
            return ""
        except Exception as e:
            logger.exception("Direct fallback OCR failed")
            OCR_ERROR_COUNT.labels(
                phase="direct_fallback_ocr", error_type=type(e).__name__
            ).inc()
            textract_text = await self.extract_text_textract(image_bytes)
            if textract_text:
                return textract_text
            return ""

    async def extract_text_textract(self, image_bytes: bytes) -> str:
        """Secondary fallback using AWS Textract synchronous byte analysis."""

        def _detect_lines() -> str:
            client = boto3.client("textract")
            response = client.detect_document_text(Document={"Bytes": image_bytes})
            lines = [
                block.get("Text", "").strip()
                for block in response.get("Blocks", [])
                if block.get("BlockType") == "LINE" and block.get("Text")
            ]
            return "\n".join(line for line in lines if line).strip()

        try:
            text = await asyncio.to_thread(_detect_lines)
            if text:
                logger.info("Direct fallback OCR succeeded via AWS Textract")
            return text
        except (ClientError, BotoCoreError) as e:
            logger.error("Textract fallback OCR failed: %s", e)
            OCR_ERROR_COUNT.labels(
                phase="direct_fallback_textract", error_type=type(e).__name__
            ).inc()
            return ""
        except Exception as e:
            logger.exception("Unexpected Textract fallback OCR failure")
            OCR_ERROR_COUNT.labels(
                phase="direct_fallback_textract", error_type=type(e).__name__
            ).inc()
            return ""

    async def run_reconstruction(self, ctx: DocumentContext, max_iterations: int):
        """Executes reconstruction preprocessor if enabled."""
        if (
            not ctx.use_reconstruction
            or not CapabilityProvider.is_reconstruction_available()
            or recon_process_bytes is None
        ):
            return

        start_time = time.time()
        status = "failure"
        try:
            logger.info("Executing reconstruction preprocessor pipeline")
            # process_bytes is CPU bound, run in thread
            recon_text, recon_img_bytes, recon_meta = await asyncio.to_thread(
                recon_process_bytes, ctx.image_bytes, iterations=max_iterations
            )

            if recon_img_bytes:
                try:
                    ctx.current_img = await ImageToolkit.decode_image_async(
                        recon_img_bytes
                    )
                    ctx.reconstruction_info = {
                        "preview_text": recon_text,
                        "meta": recon_meta,
                    }
                    logger.info("Using high-fidelity reconstructed source")
                except ImageToolkitError as e:
                    logger.warning(
                        "Failed to decode reconstructed image; using original. "
                        "Error: %s",
                        e,
                    )
                    OCR_ERROR_COUNT.labels(
                        phase="reconstruction_decode", error_type=type(e).__name__
                    ).inc()
            status = "success"
        except Exception as e:
            logger.exception("Reconstruction pipeline failed")
            OCR_ERROR_COUNT.labels(
                phase="reconstruction", error_type=type(e).__name__
            ).inc()
        finally:
            latency = time.time() - start_time
            OCR_RECONSTRUCTION_LATENCY.labels(status=status).observe(latency)

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
        start_time = time.time()
        status = "failure"
        method = "regions" if regions else "full_page"
        try:
            if regions:
                text = await self._extract_from_regions(img, regions)
            else:
                text = await asyncio.to_thread(
                    pytesseract.image_to_string, img, config=self.ocr_config.flags
                )
            status = "success"
            return text
        except (
            pytesseract.pytesseract.TesseractNotFoundError,
            pytesseract.pytesseract.TesseractError,
        ) as e:
            phase = (
                "extraction_tesseract_missing"
                if isinstance(e, pytesseract.pytesseract.TesseractNotFoundError)
                else "extraction_tesseract_runtime"
            )
            logger.error(
                "Tesseract unavailable/error during OCR extraction "
                "(configured=%s): %s",
                _TESSERACT_CMD or "auto",
                e,
            )
            OCR_ERROR_COUNT.labels(
                phase=phase,
                error_type=type(e).__name__,
            ).inc()
            try:
                ok, encoded = cv2.imencode(".png", img)
                if not ok:
                    return ""
                return await self.extract_text_textract(encoded.tobytes())
            except Exception:
                logger.exception(
                    "Textract fallback failed after missing tesseract binary"
                )
                OCR_ERROR_COUNT.labels(
                    phase="extraction_tesseract_missing_fallback",
                    error_type="TextractFallbackFailure",
                ).inc()
                return ""
        except Exception as e:
            logger.exception("OCR extraction failed | method=%s", method)
            OCR_ERROR_COUNT.labels(
                phase="extraction", error_type=type(e).__name__
            ).inc()
            return ""
        finally:
            latency = time.time() - start_time
            OCR_EXTRACTION_LATENCY.labels(method=method, status=status).observe(latency)

    async def _extract_from_regions(
        self, img: np.ndarray, regions: list[dict[str, Any]]
    ) -> str:
        """Performs targeted extraction on ROIs."""

        async def _extract_one(region):
            x, y, w, h = region["bbox"]
            roi = img[y : y + h, x : x + w]
            if roi.size == 0:
                return ""
            roi = ImageToolkit.prepare_roi(roi)
            try:
                return await asyncio.to_thread(
                    pytesseract.image_to_string, roi, config=self.ocr_config.flags
                )
            except Exception:
                logger.exception(
                    "Region extraction failed | bbox=%s", region.get("bbox")
                )
                return ""

        results = await asyncio.gather(*[_extract_one(r) for r in regions])
        return "\n\n".join([r.strip() for r in results if r.strip()])


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
            engine_config=self.config,
            reconstructor=reconstructor
            or (
                PixelReconstructor()
                if CapabilityProvider.is_reconstruction_available()
                else None
            ),
        )
        self.advanced_reconstructor = (
            advanced_reconstructor or AdvancedPixelReconstructor(client=self._client)
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
        start_time = time.time()
        status = "failure"
        try:
            validation_error = ImageToolkit.validate_image(
                image_bytes, self.config.max_image_size_mb
            )
            if validation_error:
                return {"error": validation_error}

            ctx = DocumentContext(
                image_bytes=image_bytes, use_reconstruction=use_reconstruction
            )

            if not await self.processor.decode_and_validate(ctx):
                logger.warning(
                    "Decode/validation failed; attempting direct Pillow OCR fallback"
                )
                fallback_text = (
                    await self.processor.extract_text_direct(image_bytes)
                ).strip()
                if fallback_text:
                    fallback_confidence = self.confidence_scorer.calculate(
                        fallback_text
                    )
                    status = "success"
                    return {
                        "text": fallback_text,
                        "confidence": fallback_confidence,
                        "iterations": [
                            {
                                "iteration": 0,
                                "method": "pillow-direct-fallback",
                                "text_length": len(fallback_text),
                                "confidence": fallback_confidence,
                            }
                        ],
                        "success": True,
                        "method": "pillow-direct-fallback",
                    }
                return {"error": "Corrupted or unsupported image format"}

            # Parallel initialization: Layout + Reconstruction
            await asyncio.gather(
                self.processor.run_reconstruction(ctx, self.config.max_iterations),
                self._analyze_layout(ctx),
            )

            # Iteration Loop
            for i in range(self.config.max_iterations):
                OCR_ITERATION_COUNT.inc()
                await self._run_iteration(ctx, i)

            status = "success"
            return self._build_response(ctx)
        except Exception as e:
            logger.exception("Error in process_image")
            OCR_ERROR_COUNT.labels(
                phase="process_image", error_type=type(e).__name__
            ).inc()
            return {"error": str(e)}
        finally:
            latency = time.time() - start_time
            OCR_ENGINE_PROCESS_IMAGE_LATENCY.labels(status=status).observe(latency)

    async def process_image_advanced(
        self, image_bytes: bytes, doc_type: Optional[str] = None
    ) -> dict[str, Any]:
        """AI-driven pipeline with contextual learning."""
        start_time = time.time()
        status = "failure"
        try:
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

            # Parallel initialization
            layout_task = self._analyze_layout(ctx)
            learning_task = self.learning_engine.get_pattern_knowledge(ctx.doc_type)

            await asyncio.gather(layout_task, learning_task)
            pattern = await learning_task

            context = (pattern or {}) | {
                "layout_type": ctx.layout_type,
                "region_count": len(ctx.layout_regions),
            }

            ai_result = await self.advanced_reconstructor.reconstruct_with_ai(
                image_bytes, context=context
            )

            if "error" in ai_result:
                logger.warning("AI reconstruction failed | Triggering fallback")
                return await self.process_image(image_bytes, use_reconstruction=True)

            extracted_text = ai_result.get("text", "")
            confidence = self.confidence_scorer.calculate(extracted_text)

            self._schedule_learning(
                ctx.doc_type,
                ai_result.get("model", "unknown"),
                ctx.layout_type,
                confidence,
            )
            status = "success"
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
        except Exception as e:
            logger.exception("Error in process_image_advanced")
            OCR_ERROR_COUNT.labels(
                phase="process_image_advanced", error_type=type(e).__name__
            ).inc()
            return {"error": str(e)}
        finally:
            latency = time.time() - start_time
            OCR_ENGINE_PROCESS_IMAGE_ADVANCED_LATENCY.labels(status=status).observe(
                latency
            )

    async def _analyze_layout(self, ctx: DocumentContext):
        """Analyzes document layout."""

        def _run():
            regions = DocumentLayoutAnalyzer.detect_regions(ctx.image_bytes)
            l_type = DocumentLayoutAnalyzer.classify_layout(regions)
            return regions, l_type

        try:
            ctx.layout_regions, ctx.layout_type = await asyncio.to_thread(_run)
        except LayoutAnalysisError as e:
            logger.warning("Layout analysis failed: %s", e)
            ctx.layout_regions = []
            ctx.layout_type = "unknown"

    async def _run_iteration(self, ctx: DocumentContext, i: int):
        """Executes a single iteration loop."""
        try:
            if ctx.current_img is None:
                raise ValueError("Context image is missing")

            thresh = self.processor.preprocess_frame(
                ctx.current_img, i, ctx.use_reconstruction
            )

            use_regions = (
                i == 1
                and ctx.best_confidence < self.config.confidence_threshold
                and len(ctx.layout_regions) > 1
            )
            text = await self.processor.extract_text(
                thresh, ctx.layout_regions if use_regions else None
            )

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

            ctx.current_img = await ImageToolkit.enhance_iteration(ctx.current_img)
        except Exception:
            logger.exception("Iteration %d failed", i + 1)
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
        """Schedules background learning task."""
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

        def _log_task_error(t):
            try:
                exc = t.exception()
                if exc:
                    logger.error("Background learning task failed: %s", exc)
            except Exception as cb_exc:
                logger.debug("Background learning callback failed: %s", cb_exc)
            self._background_tasks.discard(t)

        task.add_done_callback(_log_task_error)
