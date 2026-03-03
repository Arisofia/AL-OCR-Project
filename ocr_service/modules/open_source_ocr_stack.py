"""Open-source three-layer OCR router and fintech normalization primitives."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field

from ocr_service.modules.ocr_engine import IterativeOCREngine

DocumentStatus = Literal["OK", "PARTIAL", "FAILED"]
QualityClass = Literal["GOOD", "PARTIAL", "UNUSABLE"]
DATE_PATTERN = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
DATE_PATTERN = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
TXN_DATE_PATTERN = r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?"


class DocumentInput(BaseModel):
    """Canonical OCR input for single-document processing."""

    id: str
    bytes_or_path: bytes | str
    mime_type: Optional[str] = None
    document_type: str = "other"
    language: str = "eng"
    source: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResult(BaseModel):
    """Unified OCR output independent of underlying OCR engine."""

    id: str
    document_type: str
    engine_used: str
    raw_text: str
    structured_output: dict[str, Any] = Field(default_factory=dict)
    quality_score: float = 0.0
    status: DocumentStatus = "FAILED"
    errors: list[str] = Field(default_factory=list)
    fallback_chain: list[str] = Field(default_factory=list)
    debug_info: dict[str, Any] = Field(default_factory=dict)


@dataclass
class LayerOutput:
    """Intermediate layer output used by the router decision engine."""

    engine_used: str
    text: str
    structured_output: dict[str, Any]
    errors: list[str]
    debug_info: dict[str, Any]


class FintechQualityEvaluator:
    """Rule-based OCR quality evaluator for fintech document classes."""

    _txn_like = re.compile(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b")
    _currency = re.compile(r"\b(?:USD|EUR|GBP|MXN|COP|\$|€)\s?\d+[\d,\.]*\b")
    _account_like = re.compile(r"\b(?:acct|account|iban|clabe|iban:)\b", re.I)
    _date_like = re.compile(DATE_PATTERN)
    _id_like = re.compile(r"\b(?:id|passport|dni|ssn|tax id)\b", re.I)
    _merchant_like = re.compile(
        r"\b(?:store|market|shop|merchant|invoice|receipt)\b", re.I
    )
    _total_like = re.compile(r"\b(?:total|amount due|grand total)\b", re.I)

    def evaluate(
        self,
        raw_text: str,
        document_type: str,
        structured_output: Optional[dict[str, Any]] = None,
        engine_metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Score OCR quality and classify extraction usability by document type."""
        _ = structured_output
        _ = engine_metadata
        text = (raw_text or "").strip()
        if not text:
            return {
                "quality_score": 0.0,
                "classification": "UNUSABLE",
                "reasons": ["EMPTY_TEXT"],
            }

        alnum_ratio = self._alnum_ratio(text)
        printable_ratio = self._printable_ratio(text)
        length_score = min(len(text) / 700.0, 1.0)

        base_score = 0.35 * length_score + 0.35 * alnum_ratio + 0.30 * printable_ratio
        score = min(max(base_score, 0.0), 1.0)
        reasons: list[str] = []

        doc_type = (document_type or "other").lower().strip()
        score = self._apply_doc_type_signal_bonus(doc_type, text, score, reasons)
        self._append_common_quality_reasons(
            text=text,
            alnum_ratio=alnum_ratio,
            printable_ratio=printable_ratio,
            reasons=reasons,
        )
        classification = self._classify_score(score)

        return {
            "quality_score": float(round(score, 4)),
            "classification": classification,
            "reasons": reasons,
        }

    def _apply_doc_type_signal_bonus(
        self,
        doc_type: str,
        text: str,
        score: float,
        reasons: list[str],
    ) -> float:
        if doc_type == "bank_statement":
            hit_count = self._bank_statement_hits(text)
            if hit_count < 2:
                reasons.append("BANK_STATEMENT_SIGNALS_WEAK")
            return min(1.0, score + (0.12 * hit_count))

        if doc_type in {"loan_application", "kyc_form"}:
            hit_count = self._kyc_hits(text)
            if hit_count < 2:
                reasons.append("KYC_SIGNALS_WEAK")
            return min(1.0, score + (0.10 * hit_count))

        if doc_type in {"receipt", "invoice"}:
            hit_count = self._receipt_hits(text)
            if hit_count < 2:
                reasons.append("RECEIPT_SIGNALS_WEAK")
            return min(1.0, score + (0.09 * hit_count))

        return score

    def _bank_statement_hits(self, text: str) -> int:
        return sum(
            (
                bool(self._account_like.search(text)),
                len(self._currency.findall(text)) >= 2,
                len(self._txn_like.findall(text)) >= 3,
            )
        )

    def _kyc_hits(self, text: str) -> int:
        return sum(
            (
                bool(re.search(r"\bname\b", text, re.I)),
                bool(self._date_like.search(text)),
                bool(re.search(r"\baddress\b", text, re.I)),
                bool(self._id_like.search(text)),
            )
        )

    def _receipt_hits(self, text: str) -> int:
        return sum(
            (
                bool(self._merchant_like.search(text)),
                bool(self._date_like.search(text)),
                bool(self._total_like.search(text)),
                bool(self._currency.search(text)),
            )
        )

    @staticmethod
    def _append_common_quality_reasons(
        *,
        text: str,
        alnum_ratio: float,
        printable_ratio: float,
        reasons: list[str],
    ) -> None:
        if printable_ratio < 0.85:
            reasons.append("LOW_PRINTABLE_RATIO")
        if alnum_ratio < 0.35:
            reasons.append("LOW_ALNUM_RATIO")
        if len(text) < 30:
            reasons.append("TEXT_TOO_SHORT")

    @staticmethod
    def _classify_score(score: float) -> QualityClass:
        if score >= 0.75:
            return "GOOD"
        if score >= 0.45:
            return "PARTIAL"
        return "UNUSABLE"

    @staticmethod
    def _alnum_ratio(text: str) -> float:
        if not text:
            return 0.0
        alnum = sum(char.isalnum() for char in text)
        return alnum / max(len(text), 1)

    @staticmethod
    def _printable_ratio(text: str) -> float:
        if not text:
            return 0.0
        printable = sum((31 < ord(char) < 127) or char in "\n\r\t" for char in text)
        return printable / max(len(text), 1)


class FintechNormalizer:
    """Normalize raw OCR outputs into stable fintech-oriented JSON schemas."""

    def normalize(
        self,
        document_type: str,
        raw_text: str,
        structured_output: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Normalize OCR text into document-specific fintech schema sections."""
        base = {
            "document_type": document_type,
            "raw_sections": self._split_sections(raw_text),
        }
        doc_type = (document_type or "other").lower().strip()
        if doc_type == "bank_statement":
            base["bank_statement"] = self._normalize_bank_statement(raw_text)
        elif doc_type in {"loan_application", "kyc_form"}:
            base["loan_application"] = self._normalize_loan_application(raw_text)
        elif doc_type in {"receipt", "invoice"}:
            base["receipt_invoice"] = self._normalize_receipt_invoice(raw_text)

        if structured_output:
            base["engine_structured"] = structured_output
        return base

    @staticmethod
    def _split_sections(text: str) -> list[str]:
        return [
            chunk.strip()
            for chunk in re.split(r"\n\s*\n", text or "")
            if chunk.strip()
        ]

    def _normalize_bank_statement(self, text: str) -> dict[str, Any]:
        account_number = self._first_match(
            text,
            r"(?i)(?:account(?:\s*number)?|acct|iban|clabe)[:\s#-]*([A-Z0-9\-]{6,34})",
        )
        currency = self._first_match(text, r"\b(USD|EUR|GBP|MXN|COP)\b")
        statement_period = self._first_match(
            text,
            r"(?i)(?:statement\s*period|period)[:\s-]*([\w\s\-/]{6,40})",
        )

        txn_rows = [
            {"raw_line": line.strip()}
            for line in text.splitlines()
            if re.search(TXN_DATE_PATTERN, line)
            and re.search(r"[-+]?\$?\d+[\d,]*\.\d{2}", line)
        ]

        return {
            "account_holder": self._first_match(
                text,
                r"(?i)(?:account\s*holder|name)[:\s-]*([A-Z][A-Za-z\s\.'-]{3,80})",
            ),
            "account_number": account_number,
            "statement_period": statement_period,
            "currency": currency,
            "transactions": txn_rows,
        }

    def _normalize_loan_application(self, text: str) -> dict[str, Any]:
        return {
            "applicant": {
                "name": self._first_match(
                    text,
                    r"(?i)(?:name)[:\s-]*([A-Z][A-Za-z\s\.'-]{3,80})",
                ),
                "dob": self._first_match(text, DATE_PATTERN),
                "address": self._first_match(
                    text,
                    r"(?i)(?:address)[:\s-]*([^\n]{8,120})",
                ),
                "id_number": self._first_match(
                    text,
                    r"(?i)(?:id\s*number|passport|dni|ssn)[:\s-]*([A-Z0-9-]{4,24})",
                ),
                "contact": self._first_match(
                    text,
                    r"(?i)(?:phone|email)[:\s-]*([^\n]{5,80})",
                ),
            },
            "employment": {
                "employer": self._first_match(
                    text,
                    r"(?i)(?:employer|company)[:\s-]*([^\n]{3,80})",
                ),
                "income": self._first_match(
                    text,
                    r"(?i)(?:income|salary)[:\s-]*([\$A-Z]{0,3}\s?\d+[\d,\.]{0,20})",
                ),
            },
            "loan_details": {
                "amount": self._first_match(
                    text,
                    (
                        r"(?i)(?:loan\s*amount|amount\s*requested)"
                        r"[:\s-]*([\$A-Z]{0,3}\s?\d+[\d,\.]{0,20})"
                    ),
                ),
                "term": self._first_match(
                    text,
                    r"(?i)(?:term|tenor)[:\s-]*([^\n]{2,30})",
                ),
                "product": self._first_match(
                    text,
                    r"(?i)(?:product|loan\s*type)[:\s-]*([^\n]{2,40})",
                ),
            },
        }

    def _normalize_receipt_invoice(self, text: str) -> dict[str, Any]:
        return {
            "merchant": self._first_match(
                text,
                r"(?i)(?:merchant|store|vendor|shop|seller)[:\s-]*([^\n]{2,80})",
            ),
            "date": self._first_match(text, DATE_PATTERN),
            "items": [
                {"description": line.strip()}
                for line in text.splitlines()
                if re.search(r"\d+[\d,]*\.\d{2}", line)
                and len(line.strip().split()) >= 2
            ][:25],
            "total_amount": self._first_match(
                text,
                (
                    r"(?i)(?:grand\s*total|total\s*amount|amount\s*due|total)"
                    r"[:\s-]*([\$A-Z]{0,3}\s?\d+[\d,\.]{0,20})"
                ),
            ),
            "taxes": self._first_match(
                text,
                r"(?i)(?:tax|vat|gst)[:\s-]*([\$A-Z]{0,3}\s?\d+[\d,\.]{0,20})",
            ),
        }

    @staticmethod
    def _first_match(text: str, pattern: str) -> Optional[str]:
        match = re.search(pattern, text or "")
        if not match:
            return None
        if match.groups():
            return (match[1] or "").strip() or None
        return (match[0] or "").strip() or None


class OpenSourceOCRRouter:
    """Three-layer OCR orchestration with quality-based escalation."""

    def __init__(
        self,
        layer1_engine: Optional[IterativeOCREngine] = None,
        layer2_engine: Optional[IterativeOCREngine] = None,
        layer3_engine: Optional[IterativeOCREngine] = None,
        evaluator: Optional[FintechQualityEvaluator] = None,
        normalizer: Optional[FintechNormalizer] = None,
    ):
        self.layer1_engine = layer1_engine or IterativeOCREngine()
        self.layer2_engine = layer2_engine or self.layer1_engine
        self.layer3_engine = layer3_engine or self.layer1_engine
        self.evaluator = evaluator or FintechQualityEvaluator()
        self.normalizer = normalizer or FintechNormalizer()

    async def process_document(self, input_doc: DocumentInput) -> DocumentResult:
        """Process one document through adaptive three-layer OCR routing."""
        payload = self._resolve_bytes(input_doc)
        fallback_chain: list[str] = []

        l1 = await self._run_layer1(payload, input_doc)
        fallback_chain.append(l1.engine_used)
        q1 = self.evaluator.evaluate(
            l1.text,
            input_doc.document_type,
            l1.structured_output,
        )
        if q1["classification"] == "GOOD":
            return self._build_result(input_doc, l1, q1, fallback_chain)

        escalate_l2 = q1["classification"] in {"PARTIAL", "UNUSABLE"}
        if not escalate_l2:
            return self._build_result(input_doc, l1, q1, fallback_chain)

        l2 = await self._run_layer2(payload, input_doc)
        fallback_chain.append(l2.engine_used)
        q2 = self.evaluator.evaluate(
            l2.text,
            input_doc.document_type,
            l2.structured_output,
        )
        if q2["classification"] == "GOOD":
            return self._build_result(input_doc, l2, q2, fallback_chain)

        if self._should_escalate_layer3(input_doc.document_type, q2["classification"]):
            l3 = await self._run_layer3(payload, input_doc)
            fallback_chain.append(l3.engine_used)
            q3 = self.evaluator.evaluate(
                l3.text,
                input_doc.document_type,
                l3.structured_output,
            )
            return self._build_result(input_doc, l3, q3, fallback_chain)

        return self._build_result(input_doc, l2, q2, fallback_chain)

    async def process_batch(self, inputs: list[DocumentInput]) -> list[DocumentResult]:
        """Process a batch of documents concurrently with bounded gather."""
        return list(
            await asyncio.gather(*[self.process_document(doc) for doc in inputs])
        )

    async def _run_layer1(self, payload: bytes, input_doc: DocumentInput) -> LayerOutput:
        result = await self.layer1_engine.process_image(
            payload,
            use_reconstruction=False,
            doc_type=input_doc.document_type,
        )
        return LayerOutput(
            engine_used="layer1:iterative_tesseract",
            text=(result.get("text") or "").strip(),
            structured_output=result,
            errors=[] if result.get("text") else ["EMPTY_RESULT"],
            debug_info={"layer": 1, "text_len": len(result.get("text") or "")},
        )

    async def _run_layer2(self, payload: bytes, input_doc: DocumentInput) -> LayerOutput:
        result = await self.layer2_engine.process_image(
            payload,
            use_reconstruction=True,
            doc_type=input_doc.document_type,
        )
        return LayerOutput(
            engine_used="layer2:layout_aware",
            text=(result.get("text") or "").strip(),
            structured_output=result,
            errors=[] if result.get("text") else ["EMPTY_RESULT"],
            debug_info={
                "layer": 2,
                "layout_type": result.get("layout_type"),
                "text_len": len(result.get("text") or ""),
            },
        )

    async def _run_layer3(self, payload: bytes, input_doc: DocumentInput) -> LayerOutput:
        result = await self.layer3_engine.process_image_advanced(
            payload,
            doc_type=input_doc.document_type,
        )
        return LayerOutput(
            engine_used="layer3:vision_llm",
            text=(result.get("text") or "").strip(),
            structured_output=result,
            errors=[] if result.get("text") else ["EMPTY_RESULT"],
            debug_info={"layer": 3, "text_len": len(result.get("text") or "")},
        )

    def _build_result(
        self,
        input_doc: DocumentInput,
        layer_out: LayerOutput,
        quality: dict[str, Any],
        fallback_chain: list[str],
    ) -> DocumentResult:
        raw_text = layer_out.text
        normalized = self.normalizer.normalize(
            input_doc.document_type,
            raw_text,
            layer_out.structured_output,
        )

        classification = quality.get("classification", "UNUSABLE")
        if classification == "GOOD":
            status: DocumentStatus = "OK"
        elif classification == "PARTIAL":
            status = "PARTIAL"
        else:
            status = "FAILED"

        errors = list(layer_out.errors)
        if status != "OK":
            errors.extend(quality.get("reasons", []))

        return DocumentResult(
            id=input_doc.id,
            document_type=input_doc.document_type,
            engine_used=layer_out.engine_used,
            raw_text=raw_text,
            structured_output=normalized,
            quality_score=float(quality.get("quality_score", 0.0)),
            status=status,
            errors=errors,
            fallback_chain=fallback_chain,
            debug_info={
                "source": input_doc.source,
                "classification": classification,
                "reasons": quality.get("reasons", []),
                **layer_out.debug_info,
            },
        )

    @staticmethod
    def _should_escalate_layer3(document_type: str, classification: str) -> bool:
        critical_docs = {
            "bank_statement",
            "loan_application",
            "kyc_form",
            "contract",
            "disclosure",
        }
        return classification in {"PARTIAL", "UNUSABLE"} and (
            (document_type or "other").lower().strip() in critical_docs
        )

    @staticmethod
    def _resolve_bytes(input_doc: DocumentInput) -> bytes:
        if isinstance(input_doc.bytes_or_path, bytes):
            return input_doc.bytes_or_path
        path = Path(str(input_doc.bytes_or_path))
        return path.read_bytes()


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Layer-agnostic preprocessing for traditional OCR engines."""
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Invalid image input")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    target_width = 1800
    _, w = gray.shape[:2]
    if 0 < w < target_width:
        scale = target_width / float(w)
        gray = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    denoised = cv2.bilateralFilter(gray, 7, 40, 40)

    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        15,
    )

    merged = cv2.bitwise_and(otsu, adaptive)
    return _deskew(merged)


def _deskew(bin_img: np.ndarray) -> np.ndarray:
    """Estimate and correct skew angle using min-area rectangle orientation."""
    points = np.column_stack(np.nonzero(bin_img < 255))
    if len(points) < 50:
        return bin_img

    angle = cv2.minAreaRect(points)[-1]
    if angle < -45:
        angle = 90 + angle
    angle = -angle

    if abs(angle) < 0.3:
        return bin_img

    h, w = bin_img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        bin_img,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
