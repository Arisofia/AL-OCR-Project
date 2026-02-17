"""
Core OCR Orchestration Engine for high-fidelity document intelligence.
Refactored into modular components for better maintainability and performance.
"""

# This engine intentionally catches broad exceptions at API and provider boundaries
# to preserve fallback behavior and service availability.
# pylint: disable=broad-exception-caught

import asyncio
import logging
import os
import re
import string
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
                normalized_np = np.array(normalized)[:, :, ::-1].copy()
                text = await asyncio.to_thread(
                    pytesseract.image_to_string,
                    normalized,
                    config=self.ocr_config.flags,
                )
            text = self.sanitize_text(text)
            return await self._rescue_ambiguous_digits(normalized_np, text)
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

    def sanitize_text(self, text: str) -> str:
        """Sanitize and validate extracted text to prevent corruption."""
        if not text or not isinstance(text, str):
            return ""

        try:
            # Ensure valid UTF-8 encoding
            text = text.encode("utf-8", errors="ignore").decode("utf-8")

            # Remove non-printable characters but keep basic punctuation
            allowed_chars = (
                string.ascii_letters
                + string.digits
                + string.punctuation
                + " \n\t"
            )
            sanitized = "".join(
                c for c in text if c in allowed_chars or ord(c) > 127
            )

            # Remove excessive whitespace
            sanitized = re.sub(r"\s+", " ", sanitized).strip()

            # Normalize OCR punctuation noise inside grouped numeric sequences
            # (e.g., "4048-. 3700 045—" -> "4048 3700 045") while preserving
            # common decimal formats like "1.250,00".
            previous = ""
            while previous != sanitized:
                previous = sanitized
                sanitized = re.sub(
                    r"(\d{3,4})\s*[-\u2013\u2014\u2212\.,:;]+\s*(\d{3,4})",
                    r"\1 \2",
                    sanitized,
                )
            sanitized = re.sub(
                r"(\d{3,4})[-\u2013\u2014\u2212\.,:;]+(?=\s|$)",
                r"\1",
                sanitized,
            )

            # Limit reasonable text length (prevent memory issues)
            if len(sanitized) > 10000:
                sanitized = sanitized[:10000] + "..."

            return sanitized

        except Exception as e:
            logger.warning("Text sanitization failed: %s", e)
            return ""

    @staticmethod
    def _digit_count(text: str) -> int:
        """Count numeric characters in a text string."""
        return sum(char.isdigit() for char in text)

    def _needs_digit_rescue(self, text: str) -> bool:
        """
        Detect ambiguous OCR output that is mostly numeric but contains
        letter/symbol artifacts (e.g., '4048 3700 04M!').
        """
        compact = re.sub(r"\s+", "", text or "")
        if not compact:
            return False

        digits = self._digit_count(compact)
        if digits < 8:
            return False

        has_ambiguous = any(char.isalpha() or char in "!|" for char in compact)
        if not has_ambiguous:
            return False

        return (digits / len(compact)) >= 0.65

    def _prepare_digit_focus_image(self, img: np.ndarray) -> np.ndarray:
        """Generate a high-contrast frame optimized for digit OCR retries."""
        if len(img.shape) == 3:
            gray = cast(
                np.ndarray,
                cv2.cvtColor(  # pylint: disable=no-member
                    img, cv2.COLOR_BGR2GRAY  # pylint: disable=no-member
                ),
            )
        else:
            gray = img

        upscaled = self.enhancer.upscale_and_smooth(gray, scale=2)
        denoised = self.enhancer.denoise(upscaled)
        return self.enhancer.apply_threshold(denoised)

    @staticmethod
    def _normalize_digit_candidate(text: str) -> str:
        """Keep only digits/spaces from numeric-focused OCR retries."""
        candidate = re.sub(r"[^0-9\s]", "", text or "")
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if not candidate:
            return ""

        compact = candidate.replace(" ", "")
        if len(compact) >= 12 and " " not in candidate:
            return " ".join(
                compact[i : i + 4] for i in range(0, len(compact), 4)
            ).strip()
        return candidate

    @staticmethod
    def _digit_candidate_score(text: str) -> tuple[int, int, int]:
        """Score candidate text by digit density while penalizing non-digits."""
        digits = sum(char.isdigit() for char in text)
        noise = sum((not char.isdigit()) and (not char.isspace()) for char in text)
        compact_len = len(text.replace(" ", ""))
        return (digits, -noise, compact_len)

    @staticmethod
    def _format_digits_like_base(digits: str, base_text: str) -> str:
        """Format recovered digits preserving base grouping when possible."""
        base_groups = re.findall(r"\d+", base_text or "")
        if base_groups and sum(len(group) for group in base_groups) == len(digits):
            offset = 0
            grouped = []
            for group in base_groups:
                grouped.append(digits[offset : offset + len(group)])
                offset += len(group)
            return " ".join(grouped).strip()
        return " ".join(digits[i : i + 4] for i in range(0, len(digits), 4)).strip()

    def _char_box_digit_rescue(self, focus_img: np.ndarray, base_text: str) -> str:
        """
        Pixel-level character rescue:
        re-read ambiguous boxes as single numeric glyphs.
        """
        box_config = f"--oem {self.ocr_config.oem} --psm 7 -l eng"
        boxes = pytesseract.image_to_boxes(focus_img, config=box_config)
        if not boxes:
            return ""

        height = focus_img.shape[0]
        width = focus_img.shape[1]
        recovered: list[str] = []
        for line in boxes.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue

            glyph = parts[0]
            try:
                x1, y1, x2, y2 = map(int, parts[1:5])
            except ValueError:
                continue

            if glyph.isdigit():
                recovered.append(glyph)
                continue

            # Convert Tesseract box coordinates (origin bottom-left) to numpy
            # crop coordinates (origin top-left).
            left = max(0, x1 - 2)
            right = min(width, x2 + 2)
            top = max(0, height - y2 - 2)
            bottom = min(height, height - y1 + 2)
            if bottom <= top or right <= left:
                continue

            roi = focus_img[top:bottom, left:right]
            if roi.size == 0:
                continue

            char_config = (
                f"--oem {self.ocr_config.oem} --psm 10 -l eng "
                "-c tessedit_char_whitelist=0123456789 "
                "-c classify_bln_numeric_mode=1"
            )
            candidate = pytesseract.image_to_string(roi, config=char_config)
            digit = re.sub(r"\D", "", candidate)
            if digit:
                recovered.append(digit[0])
                continue

            # Ignore OCR separator noise rather than preserving it as output noise.
            if glyph in {"|", "!", "I", "l", ".", ",", ":", ";", "-", "_"}:
                continue

        if not recovered:
            return ""

        compact = "".join(recovered)
        base_digits = self._digit_count(base_text)
        if len(compact) < base_digits:
            return ""
        return self._format_digits_like_base(compact, base_text)

    async def _rescue_ambiguous_digits(
        self, img: np.ndarray, base_text: str
    ) -> str:
        """
        Re-run OCR in numeric-focused mode when mixed glyphs appear.
        No prediction/Luhn completion: only OCR retries on pixels.
        """
        if not self._needs_digit_rescue(base_text):
            return base_text

        try:
            focus_img = await asyncio.to_thread(self._prepare_digit_focus_image, img)
        except Exception as e:
            logger.warning("Digit rescue preprocessing failed: %s", e)
            return base_text

        base_digits = self._digit_count(base_text)
        best_text = base_text
        best_score = self._digit_candidate_score(base_text)

        for psm in (7, 6, 11):
            config = (
                f"--oem {self.ocr_config.oem} --psm {psm} -l eng "
                "-c tessedit_char_whitelist=0123456789 "
                "-c classify_bln_numeric_mode=1"
            )
            try:
                candidate_raw = await asyncio.to_thread(
                    pytesseract.image_to_string, focus_img, config=config
                )
            except (
                pytesseract.pytesseract.TesseractNotFoundError,
                pytesseract.pytesseract.TesseractError,
            ):
                break
            except Exception as e:
                logger.debug("Digit rescue OCR pass failed (psm=%d): %s", psm, e)
                continue

            candidate = self._normalize_digit_candidate(candidate_raw)
            if not candidate:
                continue

            candidate_digits = self._digit_count(candidate)
            if candidate_digits < base_digits:
                continue

            score = self._digit_candidate_score(candidate)
            if score > best_score:
                best_text = candidate
                best_score = score

        if best_text != base_text:
            logger.info(
                "Digit rescue improved OCR output: '%s' -> '%s'",
                base_text,
                best_text,
            )
            return best_text

        # If multi-pass full-frame OCR did not improve, inspect ambiguous glyph
        # boxes one by one and re-read each box in single-digit mode.
        if self._needs_digit_rescue(best_text):
            try:
                boxed_candidate = await asyncio.to_thread(
                    self._char_box_digit_rescue, focus_img, best_text
                )
            except Exception as e:
                logger.debug("Char-box digit rescue failed: %s", e)
                boxed_candidate = ""

            if boxed_candidate:
                boxed_score = self._digit_candidate_score(boxed_candidate)
                if boxed_score >= best_score:
                    logger.info(
                        "Char-box rescue improved OCR output: '%s' -> '%s'",
                        best_text,
                        boxed_candidate,
                    )
                    return boxed_candidate
        return best_text

    async def _rescue_ambiguous_digits_from_bytes(
        self, image_bytes: bytes, text: str
    ) -> str:
        """Apply pixel-level digit rescue by decoding original image bytes."""
        if not self._needs_digit_rescue(text):
            return text

        try:
            img = await ImageToolkit.decode_image_async(image_bytes)
        except Exception as e:
            logger.warning("Digit rescue decode failed: %s", e)
            return text
        if img is None:
            return text
        return await self._rescue_ambiguous_digits(img, text)

    async def extract_text_textract(self, image_bytes: bytes) -> str:
        """
        Secondary fallback using AWS Textract:
        sync for small docs, async for large/complex docs.
        """

        # Use sync API for smaller documents (< 5MB), async for larger ones
        if len(image_bytes) < 5 * 1024 * 1024:  # 5MB threshold
            text = await self._extract_text_textract_sync(image_bytes)
        else:
            text = await self._extract_text_textract_async(image_bytes)
        return await self._rescue_ambiguous_digits_from_bytes(image_bytes, text)

    async def _extract_text_textract_sync(self, image_bytes: bytes) -> str:
        """Synchronous Textract for smaller documents."""
        def _detect_lines() -> tuple[str, int, int]:
            client = boto3.client("textract")
            response = client.detect_document_text(Document={"Bytes": image_bytes})
            blocks = response.get("Blocks", [])
            lines = []
            line_blocks = 0
            for block in blocks:
                if block.get("BlockType") == "LINE" and block.get("Text"):
                    line_blocks += 1
                    text = block.get("Text", "")
                    if text and isinstance(text, str):
                        # Sanitize and validate the text
                        try:
                            # Ensure output remains safe for downstream consumers.
                            sanitized = self.sanitize_text(text.strip())
                            if sanitized:
                                lines.append(sanitized)
                                logger.debug("Textract LINE block: '%s'", sanitized)
                        except (UnicodeDecodeError, UnicodeEncodeError) as e:
                            logger.warning("Skipping corrupted text block: %s", e)
                            continue

            result = "\n".join(lines).strip()
            # Final validation of the complete result
            result = self.sanitize_text(result)
            logger.info("Textract final result length: %d", len(result))
            return result, len(blocks), line_blocks

        try:
            text, total_blocks, line_blocks = await asyncio.to_thread(_detect_lines)
            logger.info(
                "Sync Textract OCR completed - extracted %d chars from "
                "%d LINE blocks (total=%d)",
                len(text),
                line_blocks,
                total_blocks,
            )
            if text:
                logger.info(
                    "Sync Textract OCR succeeded - text preview: %s",
                    text[:100],
                )
            else:
                logger.warning(
                    "Sync Textract OCR returned empty text - response blocks: "
                    "%d total, %d LINE blocks",
                    total_blocks,
                    line_blocks,
                )
            return text
        except (ClientError, BotoCoreError) as e:
            logger.error("Sync Textract OCR failed: %s", e)
            OCR_ERROR_COUNT.labels(
                phase="sync_fallback_textract", error_type=type(e).__name__
            ).inc()
            return ""
        except Exception as e:
            logger.exception("Unexpected sync Textract OCR failure")
            OCR_ERROR_COUNT.labels(
                phase="sync_fallback_textract", error_type=type(e).__name__
            ).inc()
            return ""

    async def _extract_text_textract_async(self, image_bytes: bytes) -> str:
        """Asynchronous Textract for large/complex documents."""

        def _start_async_detection() -> str:
            client = boto3.client("textract")

            # Start asynchronous document text detection
            response = client.start_document_text_detection(
                Document={"Bytes": image_bytes}
            )
            job_id = response["JobId"]
            logger.info("Started Textract async job: %s", job_id)

            # Poll for completion (with optimized timeout)
            max_attempts = 60  # 5 minutes max with variable intervals
            attempt = 0

            while attempt < max_attempts:
                attempt += 1

                # Poll more frequently at first (every 5s for first 2 minutes),
                # then every 10s.
                if attempt <= 24:  # First 2 minutes
                    time.sleep(5)
                else:
                    time.sleep(10)

                status_response = client.get_document_text_detection(JobId=job_id)
                status = status_response.get("JobStatus")

                if status == "SUCCEEDED":
                    # Collect all text from all pages
                    all_text = []
                    next_token = None

                    while True:
                        if next_token:
                            result_response = client.get_document_text_detection(
                                JobId=job_id, NextToken=next_token
                            )
                        else:
                            result_response = status_response

                        blocks = result_response.get("Blocks", [])
                        page_text = []
                        for block in blocks:
                            if block.get("BlockType") == "LINE" and block.get("Text"):
                                text = block.get("Text", "")
                                if text and isinstance(text, str):
                                    try:
                                        sanitized = self.sanitize_text(text.strip())
                                        if sanitized:
                                            page_text.append(sanitized)
                                            logger.debug(
                                                "Async LINE block: '%s'",
                                                sanitized,
                                            )
                                    except (
                                        UnicodeDecodeError,
                                        UnicodeEncodeError,
                                    ) as e:
                                        logger.warning(
                                            "Skipping corrupted async text block: %s",
                                            e,
                                        )
                                        continue
                        all_text.extend(page_text)

                        next_token = result_response.get("NextToken")
                        if not next_token:
                            break

                    final_text = "\n".join(line for line in all_text if line).strip()
                    # Final sanitization of the complete result
                    final_text = self.sanitize_text(final_text)
                    logger.info(
                        "Async Textract completed: %d lines from %s "
                        "(final length: %d)",
                        len(all_text),
                        job_id,
                        len(final_text),
                    )
                    return final_text

                if status == "FAILED":
                    error_message = status_response.get(
                        "StatusMessage",
                        "Unknown error",
                    )
                    logger.error("Textract async job failed: %s", error_message)
                    raise RuntimeError(f"Textract job failed: {error_message}")

                if status in ["PARTIAL_SUCCESS", "IN_PROGRESS"]:
                    logger.info(
                        "Textract job %s: %s (attempt %d/%d)",
                        job_id,
                        status,
                        attempt,
                        max_attempts,
                    )
                    continue

                logger.warning("Unexpected Textract status: %s", status)

            raise RuntimeError(
                f"Textract job timeout after {max_attempts * 10} seconds"
            )

        try:
            text = await asyncio.to_thread(_start_async_detection)
            if text:
                logger.info(
                    "Async Textract OCR succeeded - extracted %d characters, "
                    "text preview: %s",
                    len(text),
                    text[:100],
                )
            else:
                logger.warning("Async Textract OCR returned empty text")
            return text
        except (ClientError, BotoCoreError) as e:
            logger.error("Async Textract OCR failed: %s", e)
            OCR_ERROR_COUNT.labels(
                phase="async_fallback_textract", error_type=type(e).__name__
            ).inc()
            return ""
        except Exception as e:
            logger.exception("Unexpected async Textract OCR failure")
            OCR_ERROR_COUNT.labels(
                phase="async_fallback_textract", error_type=type(e).__name__
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
        """Applies iterative preprocessing with a pixel-rescue pass."""
        working = img
        if use_recon and iteration == 0 and self.reconstructor:
            rectified = self.reconstructor.remove_redactions(working)
            working = self.reconstructor.remove_color_overlay(rectified)

        if iteration == 0:
            return self.enhancer.clean_for_ocr(working)

        gray = (
            cast(
                np.ndarray,
                cv2.cvtColor(  # pylint: disable=no-member
                    working, cv2.COLOR_BGR2GRAY  # pylint: disable=no-member
                ),
            )
            if len(working.shape) == 3
            else working
        )
        if iteration == 1:
            return self.enhancer.apply_threshold(self.enhancer.sharpen(gray))

        # Later passes prioritize faint-pixel recovery over aggressiveness.
        upscaled = self.enhancer.upscale_and_smooth(gray, scale=2)
        denoised = self.enhancer.denoise(upscaled)
        return self.enhancer.apply_threshold(denoised)

    async def extract_text(
        self,
        img: np.ndarray,
        regions: Optional[list[dict[str, Any]]] = None,
        original_bytes: Optional[bytes] = None,
    ) -> str:
        """Performs OCR on the whole image or specific regions."""
        start_time = time.time()
        status = "failure"
        method = "regions" if regions else "full_page"
        try:
            if regions:
                logger.info("Extracting text from %d regions", len(regions))
                text = await self._extract_from_regions(img, regions)
            else:
                logger.info(
                    "Starting Tesseract OCR on full page (shape: %s)",
                    img.shape,
                )
                text = await asyncio.to_thread(
                    pytesseract.image_to_string, img, config=self.ocr_config.flags
                )
                # Sanitize Tesseract output
                text = self.sanitize_text(text)
                text = await self._rescue_ambiguous_digits(img, text)
                logger.info("Tesseract completed - extracted %d characters", len(text))
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
                if original_bytes:
                    return await self.extract_text_textract(original_bytes)
                # Fallback: encode processed image when original bytes are not
                # available.
                ok, encoded = cv2.imencode(".png", img)  # pylint: disable=no-member
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
                extracted = await asyncio.to_thread(
                    pytesseract.image_to_string, roi, config=self.ocr_config.flags
                )
                cleaned = self.sanitize_text(extracted)
                return await self._rescue_ambiguous_digits(roi, cleaned)
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

            await self._maybe_apply_quality_fallbacks(ctx)
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
                thresh, ctx.layout_regions if use_regions else None, ctx.image_bytes
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

    async def _maybe_apply_quality_fallbacks(self, ctx: DocumentContext) -> None:
        """
        Apply quality fallbacks when iterative Tesseract output appears low quality.

        This catches cases where Tesseract returns gibberish without raising errors.
        """
        best_text = (ctx.best_text or "").strip()
        low_confidence = ctx.best_confidence < self.config.confidence_threshold
        too_short = len(best_text) < 12
        ambiguous_digits = self.processor._needs_digit_rescue(best_text)
        if not (low_confidence or too_short or ambiguous_digits):
            return

        logger.info(
            "OCR fallback trigger (confidence=%.2f, len=%d, ambiguous_digits=%s); "
            "attempting quality fallbacks",
            ctx.best_confidence,
            len(best_text),
            ambiguous_digits,
        )
        candidates: list[tuple[str, str]] = []

        textract_text = await self.processor.extract_text_textract(ctx.image_bytes)
        textract_text = self.processor.sanitize_text(textract_text)
        if textract_text:
            candidates.append(("textract-quality-fallback", textract_text))

        direct_text = await self.processor.extract_text_direct(ctx.image_bytes)
        direct_text = self.processor.sanitize_text(direct_text)
        if direct_text:
            candidates.append(("direct-quality-fallback", direct_text))

        if not candidates:
            return

        scored = []
        for method, candidate_text in candidates:
            candidate_conf = self.confidence_scorer.calculate(candidate_text)
            scored.append((method, candidate_text, candidate_conf))

        method, selected_text, selected_conf = max(
            scored, key=lambda item: (item[2], len(item[1]))
        )
        better_confidence = selected_conf > (ctx.best_confidence + 0.02)
        better_coverage = (
            len(selected_text) > len(best_text) * 1.5
            and selected_conf >= ctx.best_confidence
        )
        selected_digits = self.processor._digit_count(selected_text)
        base_digits = self.processor._digit_count(best_text)
        digit_gain = selected_digits > base_digits
        ambiguity_resolved = (
            ambiguous_digits and not self.processor._needs_digit_rescue(selected_text)
        )
        if not (
            better_confidence
            or better_coverage
            or digit_gain
            or ambiguity_resolved
        ):
            return

        ctx.best_text = selected_text
        ctx.best_confidence = selected_conf
        ctx.iteration_history.append(
            {
                "iteration": len(ctx.iteration_history) + 1,
                "text_length": len(selected_text),
                "confidence": selected_conf,
                "method": method,
                "preview_text": (
                    f"{selected_text[:50]}..."
                    if len(selected_text) > 50
                    else selected_text
                ),
            }
        )
        logger.info(
            "Quality fallback selected (%s, confidence=%.2f, len=%d)",
            method,
            selected_conf,
            len(selected_text),
        )

    def _build_response(self, ctx: DocumentContext) -> dict[str, Any]:
        """Formats the final engine output."""
        # Final sanitization of the best text before response
        final_text = self.processor.sanitize_text(ctx.best_text)
        resp = {
            "text": final_text,
            "confidence": ctx.best_confidence,
            "iterations": ctx.iteration_history,
            "success": len(final_text) > 0,
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
