"""
Core OCR Orchestration Engine for high-fidelity document intelligence.
Refactored into modular components for better maintainability and performance.
"""

# This engine intentionally catches broad exceptions at API and provider boundaries
# to preserve fallback behavior and service availability.
# pylint: disable=broad-exception-caught,too-many-lines

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

from ..utils.card_validator import CardValidator
from .advanced_recon import AdvancedPixelReconstructor
from .confidence import ConfidenceScorer
from .document_intelligence import DocumentIntelligence
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
        self.active_doc_type = "generic"

    def set_active_doc_type(self, doc_type: Optional[str]) -> None:
        """Set active document type to adjust OCR strategy dynamically."""
        normalized = (doc_type or "generic").strip().lower()
        self.active_doc_type = normalized or "generic"

    def _is_card_doc_type(self) -> bool:
        """Return True when active OCR strategy should use card-specific tuning."""
        return self.active_doc_type in {
            "bank_card",
            "card",
            "credit_card",
            "debit_card",
        }

    def is_card_doc_type(self) -> bool:
        """Public wrapper for card-mode checks used by the orchestration layer."""
        return self._is_card_doc_type()

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
            return textract_text or ""
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
            return textract_text or ""

    @staticmethod
    def _build_card_focus_views(focus_img: np.ndarray) -> list[np.ndarray]:
        """Create full-frame and ROI views for numeric-focused card OCR."""
        focus_views: list[np.ndarray] = [focus_img]
        if len(focus_img.shape) < 2:
            return focus_views

        height = int(focus_img.shape[0])
        for start, end in ((0.25, 0.70), (0.32, 0.78), (0.40, 0.88)):
            y1 = max(0, min(height - 1, int(height * start)))
            y2 = max(y1 + 1, min(height, int(height * end)))
            roi = focus_img[y1:y2, :]
            if roi.size > 0 and roi.shape[0] >= 20:
                focus_views.append(roi)
        return focus_views

    def _digits_only_config(self, psm: int) -> str:
        return (
            f"--oem {self.ocr_config.oem} --psm {psm} -l eng "
            "-c tessedit_char_whitelist=0123456789 "
            "-c classify_bln_numeric_mode=1"
        )

    @staticmethod
    def _compact_digit_score(compact_digits: str) -> tuple[int, int, int]:
        compact_len = len(compact_digits)
        return (
            1 if compact_len >= 13 else 0,
            -abs(16 - compact_len),
            compact_len,
        )

    async def _run_compact_digit_pass(
        self,
        view: np.ndarray,
        psm: int,
    ) -> tuple[bool, str]:
        """Run one digits-only OCR pass, returning (stop_psm_loop, compact_digits)."""
        try:
            candidate_raw = await asyncio.to_thread(
                pytesseract.image_to_string,
                view,
                config=self._digits_only_config(psm),
            )
        except (
            pytesseract.pytesseract.TesseractNotFoundError,
            pytesseract.pytesseract.TesseractError,
        ):
            return True, ""
        except Exception as e:
            logger.debug(
                "Digits-only card fallback pass failed (psm=%d): %s",
                psm,
                e,
            )
            return False, ""

        compact = re.sub(r"\D", "", candidate_raw or "")
        return (False, compact) if 8 <= len(compact) <= 19 else (False, "")

    async def _collect_best_compact_digits(self, focus_views: list[np.ndarray]) -> str:
        best_compact = ""
        best_score = (-1, float("-inf"), -1)

        for view in focus_views:
            for psm in (7, 6, 11, 13):
                stop_psm_loop, compact = await self._run_compact_digit_pass(view, psm)
                if stop_psm_loop:
                    break
                if not compact:
                    continue

                score = self._compact_digit_score(compact)
                if score > best_score:
                    best_score = score
                    best_compact = compact
        return best_compact

    async def extract_text_card_digits_only(self, image_bytes: bytes) -> str:
        """
        Card-focused fallback that extracts only digit sequences from source bytes.
        Used when normal card OCR returns little/no text.
        """
        try:
            img = await ImageToolkit.decode_image_async(image_bytes)
            if img is None:
                return ""

            prepared = self._prepare_card_input_image(img)
            if len(prepared.shape) == 3:
                prepared = self._remove_colored_stroke(prepared)
                prepared = self._remove_skin_occlusion(prepared)
            focus_img = await asyncio.to_thread(
                self._prepare_digit_focus_image,
                prepared,
            )
        except Exception as e:
            logger.warning("Digits-only card fallback preprocessing failed: %s", e)
            return ""

        focus_views = self._build_card_focus_views(focus_img)
        best_compact = await self._collect_best_compact_digits(focus_views)
        if not best_compact:
            return ""
        return " ".join(
            best_compact[i : i + 4] for i in range(0, len(best_compact), 4)
        ).strip()

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

            # Normalize OCR punctuation noise inside long digit sequences
            # (likely IDs/cards). This is intentionally conservative:
            # it only applies when the sequence contains 11-19 digits.
            def _normalize_long_digit_span(match: re.Match[str]) -> str:
                span = match.group(0)
                span = re.sub(
                    "[-\u2013\u2014\u2212\\\\/|\\.,:;'\u2018\u2019\"\u201c\u201d]+",
                    " ",
                    span,
                )
                return re.sub(r"\s+", " ", span).strip()

            sanitized = re.sub(
                "(?:\\d[-\u2013\u2014\u2212\\\\/|\\.,:;'\u2018\u2019\"\u201c\u201d\\s]*){11,19}",
                _normalize_long_digit_span,
                sanitized,
            )

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
                sanitized = f"{sanitized[:10000]}..."

            return sanitized

        except Exception as e:
            logger.warning("Text sanitization failed: %s", e)
            return ""

    @staticmethod
    def _digit_count(text: str) -> int:
        """Count numeric characters in a text string."""
        return sum(char.isdigit() for char in text)

    def digit_count(self, text: str) -> int:
        """Public wrapper for digit counting utility."""
        return self._digit_count(text)

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
        return has_ambiguous and (digits / len(compact)) >= 0.65

    def needs_digit_rescue(self, text: str) -> bool:
        """Public wrapper for ambiguous digit rescue checks."""
        return self._needs_digit_rescue(text)

    def _prepare_digit_focus_image(self, img: np.ndarray) -> np.ndarray:
        """
        Generate a high-contrast frame optimized for digit OCR retries.
        Uses 3x upscaling and CLAHE to bring out faint pixel remnants.
        """
        if len(img.shape) == 3:
            gray = cast(
                np.ndarray,
                cv2.cvtColor(  # pylint: disable=no-member
                    img, cv2.COLOR_BGR2GRAY  # pylint: disable=no-member
                ),
            )
        else:
            gray = img

        # 3x upscale for higher fidelity on small/obscured digits
        height, width = gray.shape[:2]
        upscaled = cv2.resize(
            gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC
        )

        # Contrast Limited Adaptive Histogram Equalization (CLAHE)
        # Specifically targets faint pixel remnants under thin occlusions.
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(upscaled)

        denoised = self.enhancer.denoise(enhanced)
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
    def _extract_box_roi(
        focus_img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        padding: int = 2,
    ) -> Optional[np.ndarray]:
        """Convert Tesseract box coordinates to a cropped numpy ROI."""
        height = focus_img.shape[0]
        width = focus_img.shape[1]
        left = max(0, x1 - padding)
        right = min(width, x2 + padding)
        top = max(0, height - y2 - padding)
        bottom = min(height, height - y1 + padding)
        if bottom <= top or right <= left:
            return None
        roi = focus_img[top:bottom, left:right]
        return None if roi.size == 0 else roi

    @staticmethod
    def _roi_ink_ratio(roi: np.ndarray) -> float:
        """Estimate foreground (ink) ratio in a glyph ROI."""
        if len(roi.shape) == 3:
            gray = cast(
                np.ndarray,
                cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),  # pylint: disable=no-member
            )
        else:
            gray = roi

        _, binary = cv2.threshold(  # pylint: disable=no-member
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,  # pylint: disable=no-member
        )
        return float(np.mean(binary < 200))

    def _read_single_digit(self, roi: np.ndarray) -> str:
        """Read one numeric glyph with confidence/shape safeguards."""
        ink_ratio = self._roi_ink_ratio(roi)
        if ink_ratio < 0.02:
            return ""

        char_config = (
            f"--oem {self.ocr_config.oem} --psm 10 -l eng "
            "-c tessedit_char_whitelist=0123456789 "
            "-c classify_bln_numeric_mode=1"
        )
        best_digit, best_conf = self._read_best_digit_from_data(roi, char_config)
        if not best_digit:
            best_digit, best_conf = self._fallback_single_digit(roi, char_config)
        if not best_digit:
            return ""

        h, w = roi.shape[:2]
        aspect_ratio = (w / float(h)) if h else 0.0
        if self._should_reject_single_digit(
            best_digit=best_digit,
            best_conf=best_conf,
            ink_ratio=ink_ratio,
            aspect_ratio=aspect_ratio,
        ):
            return ""

        return best_digit

    @staticmethod
    def _fallback_single_digit(roi: np.ndarray, char_config: str) -> tuple[str, float]:
        candidate = pytesseract.image_to_string(roi, config=char_config)
        digit = re.sub(r"\D", "", candidate)
        return (digit, 40.0) if len(digit) == 1 else ("", -1.0)

    @staticmethod
    def _read_best_digit_from_data(
        roi: np.ndarray,
        char_config: str,
    ) -> tuple[str, float]:
        data = pytesseract.image_to_data(
            roi,
            config=char_config,
            output_type=pytesseract.Output.DICT,
        )
        best_digit = ""
        best_conf = -1.0
        for raw_text, raw_conf in zip(data.get("text", []), data.get("conf", [])):
            digit = re.sub(r"\D", "", raw_text or "")
            if len(digit) != 1:
                continue
            try:
                conf = float(raw_conf)
            except (TypeError, ValueError):
                continue
            if conf > best_conf:
                best_digit = digit
                best_conf = conf
        return best_digit, best_conf

    @staticmethod
    def _should_reject_single_digit(
        *,
        best_digit: str,
        best_conf: float,
        ink_ratio: float,
        aspect_ratio: float,
    ) -> bool:
        if best_digit == "0":
            return best_conf < 65.0 or ink_ratio < 0.08 or aspect_ratio < 0.35
        return best_conf < 45.0

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

    def _char_box_digit_rescue(
        self,
        focus_img: np.ndarray,
        base_text: str,
        allow_digit_drop: bool = False,
    ) -> str:
        """
        Pixel-level character rescue:
        re-read ambiguous boxes as single numeric glyphs.
        """
        box_config = f"--oem {self.ocr_config.oem} --psm 7 -l eng"
        boxes = pytesseract.image_to_boxes(focus_img, config=box_config)
        if not boxes:
            return ""

        recovered: list[str] = []
        for line in boxes.splitlines():
            parsed = self._parse_box_line(line)
            if parsed is None:
                continue
            glyph, x1, y1, x2, y2 = parsed

            roi = self._extract_box_roi(focus_img, x1, y1, x2, y2)
            if roi is None:
                continue

            if digit := self._recover_digit_from_glyph(glyph, roi):
                recovered.append(digit)

        if not recovered:
            return ""

        compact = "".join(recovered)
        base_digits = self._digit_count(base_text)
        if len(compact) < base_digits:
            if not allow_digit_drop:
                return ""
            # Card mode can drop up to one dubious trailing glyph if re-reading
            # indicates it was likely OCR noise.
            if len(compact) < max(8, base_digits - 1):
                return ""
        if len(compact) < 8:
            return ""
        return self._format_digits_like_base(compact, base_text)

    @staticmethod
    def _parse_box_line(line: str) -> Optional[tuple[str, int, int, int, int]]:
        parts = line.split()
        if len(parts) < 5:
            return None
        glyph = parts[0]
        try:
            x1, y1, x2, y2 = map(int, parts[1:5])
        except ValueError:
            return None
        return glyph, x1, y1, x2, y2

    def _recover_digit_from_glyph(self, glyph: str, roi: np.ndarray) -> str:
        if glyph.isdigit():
            if glyph != "0":
                return glyph
            confirmed = self._read_single_digit(roi)
            return confirmed or ""

        if digit := self._read_single_digit(roi):
            return digit
        return ""

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
        best_text, best_score = await self._run_rescue_digit_passes(
            focus_img=focus_img,
            base_digits=base_digits,
            initial_text=best_text,
            initial_score=best_score,
        )

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

    async def _run_rescue_digit_passes(
        self,
        *,
        focus_img: np.ndarray,
        base_digits: int,
        initial_text: str,
        initial_score: tuple[int, int, int],
    ) -> tuple[str, tuple[int, int, int]]:
        best_text = initial_text
        best_score = initial_score
        for psm in (7, 6, 11):
            candidate = await self._run_single_rescue_pass(focus_img, psm)
            if not candidate:
                if candidate is None:
                    break
                continue
            if self._digit_count(candidate) < base_digits:
                continue
            score = self._digit_candidate_score(candidate)
            if score > best_score:
                best_text = candidate
                best_score = score
        return best_text, best_score

    async def _run_single_rescue_pass(
        self,
        focus_img: np.ndarray,
        psm: int,
    ) -> Optional[str]:
        config = (
            f"--oem {self.ocr_config.oem} --psm {psm} -l eng "
            "-c tessedit_char_whitelist=0123456789 "
            "-c classify_bln_numeric_mode=1"
        )
        try:
            candidate_raw = await asyncio.to_thread(
                pytesseract.image_to_string,
                focus_img,
                config=config,
            )
        except (
            pytesseract.pytesseract.TesseractNotFoundError,
            pytesseract.pytesseract.TesseractError,
        ):
            return None
        except Exception as e:
            logger.debug("Digit rescue OCR pass failed (psm=%d): %s", psm, e)
            return ""
        return self._normalize_digit_candidate(candidate_raw)

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
        return text if img is None else await self._rescue_ambiguous_digits(img, text)

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
        try:
            text, total_blocks, line_blocks = await asyncio.to_thread(
                self._detect_textract_lines,
                image_bytes,
            )
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

    def _detect_textract_lines(self, image_bytes: bytes) -> tuple[str, int, int]:
        client_factory = cast(Any, boto3.client)
        client = client_factory("textract")
        response = client.detect_document_text(Document={"Bytes": image_bytes})
        blocks = response.get("Blocks", [])
        lines: list[str] = []
        line_blocks = 0
        for block in blocks:
            sanitized = self._extract_sanitized_line_from_block(block)
            if sanitized is None:
                continue
            line_blocks += 1
            lines.append(sanitized)
            logger.debug("Textract LINE block: '%s'", sanitized)

        result = self.sanitize_text("\n".join(lines).strip())
        logger.info("Textract final result length: %d", len(result))
        return result, len(blocks), line_blocks

    def _extract_sanitized_line_from_block(self, block: Any) -> Optional[str]:
        if block.get("BlockType") != "LINE" or not block.get("Text"):
            return None

        text = block.get("Text", "")
        if not isinstance(text, str):
            return None

        try:
            sanitized = self.sanitize_text(text.strip())
        except (UnicodeDecodeError, UnicodeEncodeError) as e:
            logger.warning("Skipping corrupted text block: %s", e)
            return None
        return sanitized or None

    async def _extract_text_textract_async(self, image_bytes: bytes) -> str:
        """Byte-input fallback: async Textract requires S3 DocumentLocation."""
        logger.info(
            "Async Textract requires S3 DocumentLocation; falling back to sync"
        )
        return await self._extract_text_textract_sync(image_bytes)

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

    def _remove_skin_occlusion(self, img: np.ndarray) -> np.ndarray:
        """
        Remove likely finger/skin occlusion regions for card-like documents.
        This is conservative and never guesses missing digits.
        """
        if len(img.shape) != 3:
            return img

        hsv = cast(
            np.ndarray,
            cv2.cvtColor(  # pylint: disable=no-member
                img, cv2.COLOR_BGR2HSV  # pylint: disable=no-member
            ),
        )
        lower_skin = np.array([0, 40, 80], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(  # pylint: disable=no-member
            hsv, lower_skin, upper_skin
        )

        ratio = float(np.count_nonzero(skin_mask)) / float(skin_mask.size or 1)
        if ratio < 0.01 or ratio > 0.60:
            return img

        cleaned = img.copy()
        cleaned[skin_mask > 0] = (255, 255, 255)
        return cleaned

    def _remove_colored_stroke(self, img: np.ndarray) -> np.ndarray:
        """
        Detect colored marker/pen strokes (green, yellow, blue, red) and use
        TELEA inpainting to reconstruct underlying pixels from surrounding
        context rather than simply blanking with background color.
        Operates only when the stroke covers 0.5%-60% of the image area.
        """
        if len(img.shape) != 3:
            return img

        hsv = cast(
            np.ndarray,
            cv2.cvtColor(img, cv2.COLOR_BGR2HSV),  # pylint: disable=no-member
        )

        green_lo, green_hi = np.array([35, 80, 60]), np.array([85, 255, 255])
        yellow_lo, yellow_hi = np.array([18, 120, 100]), np.array([35, 255, 255])
        blue_lo, blue_hi = np.array([95, 80, 60]), np.array([135, 255, 255])
        red_lo1, red_hi1 = np.array([0, 100, 80]), np.array([12, 255, 255])
        red_lo2, red_hi2 = np.array([158, 100, 80]), np.array([180, 255, 255])

        combined = np.zeros(img.shape[:2], dtype=np.uint8)
        for lo, hi in (
            (green_lo, green_hi),
            (yellow_lo, yellow_hi),
            (blue_lo, blue_hi),
            (red_lo1, red_hi1),
            (red_lo2, red_hi2),
        ):
            combined = cv2.bitwise_or(  # pylint: disable=no-member
                combined,
                cv2.inRange(hsv, lo, hi),  # pylint: disable=no-member
            )

        ratio = float(np.count_nonzero(combined)) / float(combined.size or 1)
        if ratio < 0.005 or ratio > 0.60:
            return img

        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.dilate(  # pylint: disable=no-member
            combined, kernel, iterations=1
        )

        # Double-pass inpainting: TELEA for texture, NS for structural smooth
        telea = cv2.inpaint(  # pylint: disable=no-member
            img, combined, 7, cv2.INPAINT_TELEA
        )
        return cast(
            np.ndarray,
            cv2.inpaint(  # pylint: disable=no-member
                telea, combined, 3, cv2.INPAINT_NS
            ),
        )

    def preprocess_frame(
        self, img: np.ndarray, iteration: int, use_recon: bool
    ) -> np.ndarray:
        """Applies iterative preprocessing with a pixel-rescue pass."""
        working = img
        if self._is_card_doc_type():
            working = self._remove_colored_stroke(working)
            working = self._remove_skin_occlusion(working)

        if use_recon and iteration == 0 and self.reconstructor:
            rectified = self.reconstructor.remove_redactions(working)
            working = self.reconstructor.remove_color_overlay(rectified)

        if self._is_card_doc_type():
            # Pad before thresholding/morphology so edge-touching digits are not lost.
            try:
                h, w = working.shape[:2]
                pad = min(32, max(8, int(round(0.02 * max(h, w)))))
                working = ImageToolkit.prepare_roi(working, padding=pad)
            except Exception as exc:
                logger.debug("Card-mode preprocess padding failed: %s", exc)

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

    def _card_ocr_configs(self) -> list[str]:
        """Build Tesseract configs tuned for payment card documents."""
        configs = [
            (
                f"--oem {self.ocr_config.oem} --psm 6 -l eng "
                "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/- "
                "-c classify_bln_numeric_mode=1"
            ),
            (
                f"--oem {self.ocr_config.oem} --psm 7 -l eng "
                "-c tessedit_char_whitelist=0123456789/- "
                "-c classify_bln_numeric_mode=1"
            ),
            (
                f"--oem {self.ocr_config.oem} --psm 11 -l eng "
                "-c tessedit_char_whitelist=0123456789/- "
                "-c classify_bln_numeric_mode=1"
            ),
        ]
        limit = max(1, int(self.engine_config.card_ocr_pass_limit))
        return configs[:limit]

    @staticmethod
    def _last_group_is_truncated(text: str) -> bool:
        """
        Return True when the last numeric group looks like a partial capture
        (1-3 digits), common in cropped card snapshots.
        """
        groups = re.findall(r"\d+", text or "")
        return bool(groups) and len(groups[-1]) in {1, 2, 3}

    def _score_card_text(
        self, text: str
    ) -> tuple[int, int, int, int, int, int, int]:
        """Score card OCR candidates preferring valid Luhn and cleaner digits."""
        analysis = DocumentIntelligence.analyze(text)
        card_analysis = analysis.get("card_analysis", {}) if analysis else {}
        candidates = card_analysis.get("candidates", [])
        plausible_count = sum(row.get("length", 0) >= 13 for row in candidates)
        max_len = max((row.get("length", 0) for row in candidates), default=0)
        valid_count = card_analysis.get("luhn_valid_count", 0)
        truncated_bonus = (
            1 if max_len < 13 and self._last_group_is_truncated(text) else 0
        )
        suspicious_tail_penalty = (
            1 if self._has_suspicious_partial_zero_tail(text) else 0
        )
        digit_count = self._digit_count(text)
        noise = sum((not char.isdigit()) and (not char.isspace()) for char in text)
        return (
            valid_count,
            plausible_count,
            truncated_bonus,
            -suspicious_tail_penalty,
            max_len,
            digit_count,
            -noise,
        )

    def score_card_text(self, text: str) -> tuple[int, int, int, int, int, int, int]:
        """Public wrapper for card candidate scoring."""
        return self._score_card_text(text)

    def _has_suspicious_partial_zero_tail(self, text: str) -> bool:
        """
        Detect likely spurious trailing zero in short/partial card-like strings,
        such as '4048 3700 0450', where the final glyph is low certainty.
        """
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned or not cleaned.endswith("0"):
            return False

        groups = re.findall(r"\d+", cleaned)
        if not groups or not groups[-1].endswith("0"):
            return False

        compact = "".join(groups)
        if len(compact) < 9 or len(compact) >= 13:
            return False

        analysis = DocumentIntelligence.analyze(cleaned)
        card_analysis = analysis.get("card_analysis", {})
        if card_analysis.get("luhn_valid_count", 0) > 0:
            return False

        if not (candidates := card_analysis.get("candidates", [])):
            return False
        return all(row.get("length", 0) < 13 for row in candidates)

    def _trim_spurious_trailing_zero_variant(self, text: str) -> str:
        """
        Build a conservative variant that drops one trailing zero when OCR likely
        over-read an incomplete card tail (e.g. "4048 3700 0450").
        """
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned or not cleaned.endswith("0"):
            return ""

        groups = re.findall(r"\d+", cleaned)
        if len(groups) < 2 or len(groups[-1]) != 4:
            return ""

        compact = "".join(groups)
        if len(compact) < 9 or len(compact) >= 13:
            return ""

        analysis = DocumentIntelligence.analyze(cleaned)
        card_analysis = analysis.get("card_analysis", {})
        if card_analysis.get("luhn_valid_count", 0) > 0:
            return ""

        candidates = card_analysis.get("candidates", [])
        if not candidates:
            return ""
        if any(row.get("length", 0) >= 13 for row in candidates):
            return ""

        trimmed = compact[:-1]
        if len(trimmed) < 8:
            return ""
        return self._format_digits_like_base(trimmed, cleaned)

    def _mark_uncertain_partial_card_tail(self, text: str) -> str:
        """
        Avoid returning a false trailing zero for partial card-like captures.
        Instead, mark the last glyph as uncertain for manual review.
        """
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not self._has_suspicious_partial_zero_tail(cleaned):
            return cleaned
        return re.sub(r"0$", "?", cleaned)

    def mark_uncertain_partial_card_tail(self, text: str) -> str:
        """Public wrapper to normalize uncertain trailing partial PAN glyphs."""
        return self._mark_uncertain_partial_card_tail(text)

    def _prepare_card_input_image(self, img: np.ndarray) -> np.ndarray:
        """Apply conservative border padding for card OCR."""
        try:
            h, w = img.shape[:2]
            pad = min(32, max(8, int(round(0.02 * max(h, w)))))
            return ImageToolkit.prepare_roi(img, padding=pad)
        except Exception as exc:
            logger.debug("Card-mode border padding failed: %s", exc)
            return img

    def _card_strategy_images(self, img: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Build candidate strategy images for card OCR."""
        strategies: list[tuple[str, np.ndarray]] = [("raw", img)]
        if len(img.shape) != 3:
            return strategies

        cleaned = self._remove_skin_occlusion(img)
        if not np.array_equal(cleaned, img):
            strategies.append(("skin-cleaned", cleaned))
        return strategies

    async def _prepare_card_focus_image(
        self, work_img: np.ndarray
    ) -> Optional[np.ndarray]:
        """Prepare numeric-focused image used for box-level rescue."""
        try:
            return await asyncio.to_thread(self._prepare_digit_focus_image, work_img)
        except Exception as e:
            logger.debug("Card-mode focus preprocessing failed: %s", e)
            return None

    async def _append_card_candidate_variants(
        self,
        candidates: list[tuple[str, str]],
        prefix: str,
        label: str,
        candidate_text: str,
        focus_img: Optional[np.ndarray],
    ) -> None:
        """Append base/trimmed/box-rescued variants for a card OCR candidate."""
        if not candidate_text:
            return

        full_label = f"{prefix}-{label}"
        candidates.append((full_label, candidate_text))

        trimmed_variant = self._trim_spurious_trailing_zero_variant(candidate_text)
        if trimmed_variant and trimmed_variant != candidate_text:
            candidates.append((f"{full_label}-trim", trimmed_variant))

        if focus_img is None:
            return

        try:
            boxed_candidate = await asyncio.to_thread(
                self._char_box_digit_rescue,
                focus_img,
                candidate_text,
                True,
            )
        except Exception as exc:
            logger.debug("Card OCR box rescue failed for %s: %s", full_label, exc)
            return

        if boxed_candidate and boxed_candidate != candidate_text:
            candidates.append((f"{full_label}-box", boxed_candidate))

    async def _collect_card_candidates_for_strategy(
        self,
        work_img: np.ndarray,
        prefix: str,
    ) -> list[tuple[str, str]]:
        """Collect OCR candidates for a single card strategy image."""
        candidates: list[tuple[str, str]] = []
        focus_img = await self._prepare_card_focus_image(work_img)

        base_text = await asyncio.to_thread(
            pytesseract.image_to_string,
            work_img,
            config=self.ocr_config.flags,
        )
        base_text = self.sanitize_text(base_text)
        if base_text:
            base_text = await self._rescue_ambiguous_digits(work_img, base_text)
            await self._append_card_candidate_variants(
                candidates, prefix, "default", base_text, focus_img
            )

        for idx, config in enumerate(self._card_ocr_configs(), start=1):
            try:
                candidate_text = await asyncio.to_thread(
                    pytesseract.image_to_string,
                    work_img,
                    config=config,
                )
            except (
                pytesseract.pytesseract.TesseractNotFoundError,
                pytesseract.pytesseract.TesseractError,
            ):
                raise
            except Exception as e:
                logger.debug("Card OCR pass %d failed (%s): %s", idx, prefix, e)
                continue

            candidate_text = self.sanitize_text(candidate_text)
            if not candidate_text:
                continue

            candidate_text = await self._rescue_ambiguous_digits(
                work_img, candidate_text
            )
            if candidate_text:
                await self._append_card_candidate_variants(
                    candidates,
                    prefix,
                    f"card-pass-{idx}",
                    candidate_text,
                    focus_img,
                )
        return candidates

    def _select_best_card_candidate(
        self, candidates: list[tuple[str, str]]
    ) -> tuple[str, str]:
        """Select best candidate by card-aware scoring and mark uncertain tail."""
        method, selected = max(
            candidates,
            key=lambda item: self._score_card_text(item[1]),
        )
        return method, self._mark_uncertain_partial_card_tail(selected)

    async def _extract_text_card_mode(self, img: np.ndarray) -> str:
        """
        Card-specific OCR strategy:
        run multiple numeric-focused passes and pick the best validated output.
        """
        prepared_img = self._prepare_card_input_image(img)
        strategy_images = self._card_strategy_images(prepared_img)

        all_candidates: list[tuple[str, str]] = []
        for prefix, strategy_img in strategy_images:
            all_candidates.extend(
                await self._collect_card_candidates_for_strategy(strategy_img, prefix)
            )

        if not all_candidates:
            return ""

        method, selected = self._select_best_card_candidate(all_candidates)
        logger.info(
            "Card OCR strategy selected %s (len=%d, score=%s)",
            method,
            len(selected),
            self._score_card_text(selected),
        )
        return selected

    async def _extract_text_card_timeout_fallback(
        self,
        img: np.ndarray,
        original_bytes: Optional[bytes],
    ) -> str:
        """Fallback path when card-mode OCR times out."""
        timeout_candidates: list[tuple[str, str]] = []
        if original_bytes:
            digits_only_text = await self.extract_text_card_digits_only(original_bytes)
            if digits_only_text := self.sanitize_text(digits_only_text):
                timeout_candidates.append(
                    ("card-timeout-digits-only", digits_only_text)
                )

        fast_text = await asyncio.to_thread(
            pytesseract.image_to_string,
            img,
            config=self.ocr_config.flags,
        )
        if fast_text := self.sanitize_text(fast_text):
            fast_text = await self._rescue_ambiguous_digits(img, fast_text)
            timeout_candidates.append(("card-timeout-fast", fast_text))

        if not timeout_candidates:
            return ""
        _, selected = self._select_best_card_candidate(timeout_candidates)
        return selected

    async def _extract_text_full_page(
        self,
        img: np.ndarray,
        original_bytes: Optional[bytes],
    ) -> str:
        """Extract OCR text from full page with card-aware strategy."""
        logger.info("Starting Tesseract OCR on full page (shape: %s)", img.shape)

        if self._is_card_doc_type():
            timeout_seconds = max(
                1.0,
                self.engine_config.card_ocr_timeout_seconds,
            )
            try:
                text = await asyncio.wait_for(
                    self._extract_text_card_mode(img),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Card OCR timed out after %.1fs, using fast fallback pass",
                    self.engine_config.card_ocr_timeout_seconds,
                )
                text = await self._extract_text_card_timeout_fallback(
                    img,
                    original_bytes,
                )
            return text

        text = await asyncio.to_thread(
            pytesseract.image_to_string,
            img,
            config=self.ocr_config.flags,
        )
        text = self.sanitize_text(text)
        return await self._rescue_ambiguous_digits(img, text)

    async def _extract_text_textract_fallback(
        self,
        img: np.ndarray,
        original_bytes: Optional[bytes],
    ) -> str:
        """Fallback to Textract when local Tesseract is unavailable."""
        if original_bytes:
            return await self.extract_text_textract(original_bytes)

        ok, encoded = cv2.imencode(".png", img)  # pylint: disable=no-member
        return await self.extract_text_textract(encoded.tobytes()) if ok else ""

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
                text = await self._extract_text_full_page(img, original_bytes)
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
                return await self._extract_text_textract_fallback(img, original_bytes)
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
                if self._is_card_doc_type():
                    return await self._extract_text_card_mode(roi)

                extracted = await asyncio.to_thread(
                    pytesseract.image_to_string,
                    roi,
                    config=self.ocr_config.flags,
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
        self.validator = CardValidator(
            provider=os.getenv("CARD_VALIDATOR_PROVIDER"),
            api_key=os.getenv("CARD_VALIDATOR_API_KEY")
        )
        self._background_tasks: set[asyncio.Task] = set()

    async def close(self) -> None:
        """Cleanup engine resources."""
        await self.advanced_reconstructor.close()
        await self.validator.close()

    async def process_image(
        self,
        image_bytes: bytes,
        use_reconstruction: bool = False,
        doc_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Entry point for standard iterative OCR pipeline."""
        start_time = time.time()
        status = "failure"
        try:
            if validation_error := ImageToolkit.validate_image(
                image_bytes, self.config.max_image_size_mb
            ):
                return {"error": validation_error}

            effective_doc_type = doc_type or self.config.default_doc_type
            effective_reconstruction = self.config.effective_use_reconstruction(
                use_reconstruction,
                effective_doc_type,
            )

            ctx = DocumentContext(
                image_bytes=image_bytes,
                use_reconstruction=effective_reconstruction,
                doc_type=effective_doc_type,
            )
            self.processor.set_active_doc_type(ctx.doc_type)

            if not await self.processor.decode_and_validate(ctx):
                logger.warning(
                    "Decode/validation failed; attempting direct Pillow OCR fallback"
                )
                fallback_text = (
                    await self.processor.extract_text_direct(image_bytes)
                ).strip()
                if fallback_text:
                    fallback_text = self.processor.mark_uncertain_partial_card_tail(
                        self.processor.sanitize_text(fallback_text)
                    )
                    fallback_confidence = self.confidence_scorer.calculate(
                        fallback_text
                    )
                    fallback_analysis = DocumentIntelligence.analyze(
                        fallback_text,
                        layout_type="unknown",
                        include_bin_info=self.config.enable_bin_lookup,
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
                        **fallback_analysis,
                    }
                return {"error": "Corrupted or unsupported image format"}

            # Parallel initialization: Layout + Reconstruction
            await asyncio.gather(
                self.processor.run_reconstruction(ctx, self.config.max_iterations),
                self._analyze_layout(ctx),
            )

            # Iteration loop. Card mode has a tighter budget to reduce
            # serverless timeout risk on low-CPU runtimes.
            iterations = self.config.max_iterations
            if self.processor.is_card_doc_type():
                iterations = min(
                    iterations,
                    max(1, int(self.config.max_iterations_card)),
                )

            for i in range(iterations):
                OCR_ITERATION_COUNT.inc()
                await self._run_iteration(ctx, i)
                if self.processor.is_card_doc_type():
                    card_score = self.processor.score_card_text(ctx.best_text)
                    if card_score[0] > 0 or card_score[4] >= 13:
                        logger.info(
                            "Card OCR early stop at iteration %d with score=%s",
                            i + 1,
                            card_score,
                        )
                        break

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
            if validation_error := ImageToolkit.validate_image(
                image_bytes, self.config.max_image_size_mb
            ):
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
                return await self.process_image(
                    image_bytes,
                    use_reconstruction=True,
                    doc_type=ctx.doc_type,
                )

            extracted_text = self.processor.mark_uncertain_partial_card_tail(
                self.processor.sanitize_text(ai_result.get("text", ""))
            )

            confidence = self.confidence_scorer.calculate(extracted_text)
            analysis = DocumentIntelligence.analyze(
                extracted_text,
                layout_type=ctx.layout_type,
                include_bin_info=self.config.enable_bin_lookup,
            )

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
                **analysis,
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

            self.processor.set_active_doc_type(ctx.doc_type)

            ocr_input = self.processor.preprocess_frame(
                ctx.current_img, i, ctx.use_reconstruction
            )

            use_regions = (
                (
                    i == 1
                    or (i == 0 and self.config.prefers_layout_regions(ctx.doc_type))
                )
                and ctx.best_confidence < self.config.confidence_threshold
                and len(ctx.layout_regions) > 1
            )
            text = await self.processor.extract_text(
                ocr_input,
                ctx.layout_regions if use_regions else None,
                ctx.image_bytes,
            )

            confidence = self.confidence_scorer.calculate(text)
            ctx.iteration_history.append(
                {
                    "iteration": i + 1,
                    "text_length": len(text),
                    "confidence": confidence,
                    "method": "region-based" if use_regions else "full-page",
                    "preview_text": f"{text[:50]}..." if len(text) > 50 else text,
                    "doc_type": ctx.doc_type,
                }
            )

            if ctx.doc_type in {"generic", "unknown"}:
                detected_type = DocumentIntelligence.analyze(
                    text,
                    layout_type=ctx.layout_type,
                ).get("document_type", "generic")
                if detected_type == "bank_card":
                    ctx.doc_type = "bank_card"
                    self.processor.set_active_doc_type(ctx.doc_type)
                    ctx.iteration_history[-1]["doc_type"] = ctx.doc_type
                    logger.info(
                        "Auto-detected document type as bank_card at iteration %d",
                        i + 1,
                    )

            if confidence > ctx.best_confidence:
                ctx.best_text, ctx.best_confidence = text, confidence

            ctx.current_img = await ImageToolkit.enhance_iteration(ctx.current_img)
        except Exception:
            logger.exception("Iteration %d failed", i + 1)
            ctx.iteration_history.append({"iteration": i + 1, "error": "failed"})

    async def _extract_text_multimodal_fallback(self, ctx: DocumentContext) -> str:
        """
        Vision-LLM quality fallback. It is constrained to extraction only:
        no guessing/completion of hidden digits.
        """
        providers = getattr(self.advanced_reconstructor, "providers", {})
        if not providers:
            return ""

        best_text = (ctx.best_text or "").strip()
        best_digit_count = self.processor.digit_count(best_text)
        is_card_empty = self.processor.is_card_doc_type() and best_digit_count < 8
        if is_card_empty:
            strict_rules = (
                "This is a payment card image with a colored stroke partially covering "
                "the card number. Look extremely closely at the pixel remnants and "
                "partial digit shapes visible at the edges of and underneath the "
                "colored overlay, particularly at positions 9 and 12 (the start and "
                "end of the third digit group). "
                "Extract every digit or partial digit you can infer from visible "
                "pixels. "
                "For each digit position you cannot determine from visible pixels, "
                "output '?'. Never fabricate digits with no pixel evidence. "
                "Report the card number in groups of 4 (e.g. 4388 54?? ???? 0665). "
                "Also extract expiry date and cardholder name if visible."
            )
        else:
            strict_rules = (
                "Return only text that is visibly present in the image. "
                "Do not infer, complete, or guess occluded characters. "
                "For card-like numbers, if a character is uncertain, output '?'. "
                "Never generate missing PAN digits or check digits."
            )
        context = {
            "layout_type": ctx.layout_type,
            "doc_type": ctx.doc_type,
            "strict_instructions": strict_rules,
        }

        try:
            # Send the preprocessed image (with colored stroke removed) to the LLM
            # instead of the raw original for significantly better fidelity.
            multimodal_img_bytes = ctx.image_bytes
            if ctx.current_img is not None:
                preprocessed_bytes = await ImageToolkit.encode_image_async(
                    ctx.current_img
                )
                if preprocessed_bytes:
                    multimodal_img_bytes = preprocessed_bytes

            ai_result = await self.advanced_reconstructor.reconstruct_with_ai(
                multimodal_img_bytes,
                context=context,
                fallback=True,
            )
        except Exception as e:
            logger.warning("Vision LLM fallback invocation failed: %s", e)
            return ""

        if not isinstance(ai_result, dict):
            return ""
        if "error" in ai_result:
            logger.info("Vision LLM fallback unavailable: %s", ai_result.get("error"))
            return ""

        text = self.processor.sanitize_text(ai_result.get("text", ""))
        return self.processor.mark_uncertain_partial_card_tail(text)

    def _quality_fallback_state(
        self, ctx: DocumentContext
    ) -> tuple[str, bool, bool, bool]:
        """Compute fallback trigger state from current best OCR output."""
        best_text = (ctx.best_text or "").strip()
        low_confidence = ctx.best_confidence < self.config.confidence_threshold
        too_short = len(best_text) < 12
        ambiguous_digits = self.processor.needs_digit_rescue(best_text)
        return best_text, low_confidence, too_short, ambiguous_digits

    async def _collect_quality_fallback_candidates(
        self,
        ctx: DocumentContext,
        is_card_mode: bool,
        too_short: bool,
    ) -> list[tuple[str, str]]:
        """Collect candidate texts from fallback providers."""
        candidates: list[tuple[str, str]] = []
        if is_card_mode:
            digit_text = await self.processor.extract_text_card_digits_only(
                ctx.image_bytes
            )
            if digit_text := self.processor.sanitize_text(digit_text):
                candidates.append(("digits-only-quality-fallback", digit_text))

        if not is_card_mode:
            textract_text = await self.processor.extract_text_textract(ctx.image_bytes)
            if textract_text := self.processor.sanitize_text(textract_text):
                candidates.append(("textract-quality-fallback", textract_text))

        direct_text = await self.processor.extract_text_direct(ctx.image_bytes)
        direct_text = self.processor.sanitize_text(direct_text)
        if direct_text:
            candidates.append(("direct-quality-fallback", direct_text))

        direct_text_digit_count = self.processor.digit_count(direct_text)
        direct_text_weak = not direct_text or direct_text_digit_count < 8
        needs_vision_fallback = not is_card_mode or (
            too_short and direct_text_weak
        )
        if (
            needs_vision_fallback
            and self.config.allows_vision_quality_fallback(ctx.doc_type)
        ):
            vision_text = await self._extract_text_multimodal_fallback(ctx)
            if vision_text:
                candidates.append(("vision-llm-quality-fallback", vision_text))
        return candidates

    def _select_best_quality_candidate(
        self,
        candidates: list[tuple[str, str]],
        is_card_mode: bool = False,
    ) -> tuple[str, str, float]:
        """Select fallback candidate using confidence and card-aware ranking."""
        scored = [
            (method, candidate_text, self.confidence_scorer.calculate(candidate_text))
            for method, candidate_text in candidates
        ]
        if is_card_mode:
            card_scored = [
                (
                    method,
                    candidate_text,
                    confidence,
                    self.processor.score_card_text(candidate_text),
                    self.processor.digit_count(candidate_text),
                )
                for method, candidate_text, confidence in scored
            ]
            method, candidate_text, confidence, _, _ = max(
                card_scored,
                key=lambda item: (item[3], item[4], item[2], len(item[1])),
            )
            return method, candidate_text, confidence
        return max(scored, key=lambda item: (item[2], len(item[1])))

    def _quality_candidate_is_better(
        self,
        best_text: str,
        best_confidence: float,
        selected_text: str,
        selected_conf: float,
        ambiguous_digits: bool,
        is_card_mode: bool = False,
    ) -> bool:
        """Decide whether fallback candidate should replace current best output."""
        better_confidence = selected_conf > (best_confidence + 0.02)
        better_coverage = (
            not is_card_mode
            and len(selected_text) > len(best_text) * 1.5
            and selected_conf >= best_confidence
        )
        selected_digits = self.processor.digit_count(selected_text)
        base_digits = self.processor.digit_count(best_text)
        digit_gain = selected_digits > base_digits
        better_card_score = False
        if is_card_mode:
            better_card_score = (
                self.processor.score_card_text(selected_text)
                > self.processor.score_card_text(best_text)
            )
        partial_card_recovery = (
            is_card_mode and selected_digits >= 8 and base_digits < 8
        )
        ambiguity_resolved = (
            ambiguous_digits and not self.processor.needs_digit_rescue(selected_text)
        )
        return (
            better_confidence
            or better_coverage
            or digit_gain
            or better_card_score
            or partial_card_recovery
            or ambiguity_resolved
        )

    @staticmethod
    def _fallback_preview_text(text: str) -> str:
        return f"{text[:50]}..." if len(text) > 50 else text

    def _apply_quality_fallback_selection(
        self,
        ctx: DocumentContext,
        method: str,
        selected_text: str,
        selected_conf: float,
    ) -> None:
        """Apply selected fallback candidate to engine context."""
        ctx.best_text = selected_text
        ctx.best_confidence = selected_conf
        ctx.iteration_history.append(
            {
                "iteration": len(ctx.iteration_history) + 1,
                "text_length": len(selected_text),
                "confidence": selected_conf,
                "method": method,
                "preview_text": self._fallback_preview_text(selected_text),
            }
        )

    async def _maybe_apply_quality_fallbacks(self, ctx: DocumentContext) -> None:
        """
        Apply quality fallbacks when iterative Tesseract output appears low quality.

        This catches cases where Tesseract returns gibberish without raising errors.
        """
        best_text, low_confidence, too_short, ambiguous_digits = (
            self._quality_fallback_state(ctx)
        )
        if self.processor.is_card_doc_type():
            card_score = self.processor.score_card_text(best_text)
            if card_score[0] > 0 or card_score[4] >= 13:
                return
        if not (low_confidence or too_short or ambiguous_digits):
            return

        logger.info(
            "OCR fallback trigger (confidence=%.2f, len=%d, ambiguous_digits=%s); "
            "attempting quality fallbacks",
            ctx.best_confidence,
            len(best_text),
            ambiguous_digits,
        )
        is_card_mode = self.processor.is_card_doc_type()
        candidates = await self._collect_quality_fallback_candidates(
            ctx=ctx,
            is_card_mode=is_card_mode,
            too_short=too_short,
        )
        if not candidates:
            return

        method, selected_text, selected_conf = self._select_best_quality_candidate(
            candidates,
            is_card_mode=is_card_mode,
        )
        if not self._quality_candidate_is_better(
            best_text=best_text,
            best_confidence=ctx.best_confidence,
            selected_text=selected_text,
            selected_conf=selected_conf,
            ambiguous_digits=ambiguous_digits,
            is_card_mode=is_card_mode,
        ):
            return

        self._apply_quality_fallback_selection(
            ctx=ctx,
            method=method,
            selected_text=selected_text,
            selected_conf=selected_conf,
        )
        logger.info(
            "Quality fallback selected (%s, confidence=%.2f, len=%d)",
            method,
            selected_conf,
            len(selected_text),
        )

    @staticmethod
    def _estimate_pixel_coverage_ratio(img: Optional[np.ndarray]) -> Optional[float]:
        """Estimate foreground pixel coverage ratio in the source document image."""
        if img is None:
            return None
        try:
            gray = (
                cast(
                    np.ndarray,
                    cv2.cvtColor(  # pylint: disable=no-member
                        img, cv2.COLOR_BGR2GRAY  # pylint: disable=no-member
                    ),
                )
                if len(img.shape) == 3
                else img
            )
            _, binary = cv2.threshold(  # pylint: disable=no-member
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,  # pylint: disable=no-member
            )
            total = float(binary.size or 1)
            foreground = float(np.count_nonzero(binary == 0))
            return round(max(0.0, min(1.0, foreground / total)), 4)
        except Exception:
            logger.debug("Failed to estimate pixel coverage ratio", exc_info=True)
            return None

    @staticmethod
    def _estimate_iteration_convergence(
        iteration_history: list[dict[str, Any]],
    ) -> Optional[float]:
        """Estimate convergence as confidence delta between first and last OCR pass."""
        confidences: list[float] = []
        for row in iteration_history:
            try:
                confidences.append(float(row.get("confidence", 0.0)))
            except (TypeError, ValueError):
                continue
        if len(confidences) < 2:
            return None
        delta = confidences[-1] - confidences[0]
        return round(delta, 4)

    @staticmethod
    def _detect_pixel_rescue_applied(
        iteration_history: list[dict[str, Any]],
    ) -> bool:
        """Detect whether any non-default fallback/rescue path was used."""
        for row in iteration_history:
            method = str(row.get("method", "")).lower()
            if any(
                token in method
                for token in ("fallback", "rescue", "box", "textract", "card-pass")
            ):
                return True
        return False

    def _build_response(self, ctx: DocumentContext) -> dict[str, Any]:
        """Formats the final engine output."""
        requested_type = (ctx.doc_type or "").strip().lower()
        requested_card_type = requested_type in {
            "bank_card",
            "card",
            "credit_card",
            "debit_card",
        }
        # Final sanitization of the best text before response
        final_text = self.processor.sanitize_text(ctx.best_text)
        final_text = self.processor.mark_uncertain_partial_card_tail(final_text)
        card_text_usable = (
            self._is_usable_card_capture(final_text) if requested_card_type else True
        )
        response_confidence = ctx.best_confidence
        if requested_card_type and not card_text_usable:
            final_text = "No card detected / recapture required"
            response_confidence = min(ctx.best_confidence, 0.20)

        analysis = DocumentIntelligence.analyze(
            final_text,
            layout_type=ctx.layout_type,
            include_bin_info=self.config.enable_bin_lookup,
        )
        resp = {
            "text": final_text,
            "confidence": response_confidence,
            "iterations": ctx.iteration_history,
            "success": card_text_usable if requested_card_type else len(final_text) > 0,
            "pixel_coverage_ratio": self._estimate_pixel_coverage_ratio(
                ctx.original_img
            ),
            "readability_index": round(max(0.0, min(1.0, response_confidence)), 4),
            "iteration_convergence": self._estimate_iteration_convergence(
                ctx.iteration_history
            ),
            "pixel_rescue_applied": self._detect_pixel_rescue_applied(
                ctx.iteration_history
            ),
            **analysis,
        }

        # Override detected type with requested type if the user was explicit.
        # We use a strict check here: if the user said it's a card, it's a card.
        if requested_type and requested_type not in {"generic", "unknown"}:
            resp["document_type"] = requested_type
            if requested_card_type and not card_text_usable:
                resp["type_confidence"] = 0.0
                card_analysis = resp.get("card_analysis")
                if not isinstance(card_analysis, dict):
                    card_analysis = {}
                card_analysis.update(
                    {
                        "detected": False,
                        "requires_manual_review": True,
                        "reason": "no_card_detected",
                    }
                )
                resp["card_analysis"] = card_analysis
            else:
                resp["type_confidence"] = 1.0  # Force maximum confidence for user intent

        if ctx.reconstruction_info:
            resp["reconstruction"] = ctx.reconstruction_info
        return resp

    def _is_usable_card_capture(self, text: str) -> bool:
        """Heuristic gate to suppress gibberish outputs in explicit card mode."""
        if not text:
            return False

        valid_count, _, _, _, max_len, digit_count, _ = self.processor.score_card_text(text)
        noise_count = sum((not ch.isdigit()) and (not ch.isspace()) for ch in text)

        if valid_count > 0 or max_len >= 13:
            return True
        if digit_count < 8:
            return False
        return (max_len >= 8 and noise_count <= 2) or (
            digit_count >= 12 and noise_count <= 3
        )

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
