"""
Pydantic schemas for the OCR service API.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel


class OCRIteration(BaseModel):
    """
    Schema for a single OCR iteration result.
    """

    iteration: int
    text_length: int
    confidence: float
    preview_text: str
    method: Optional[str] = "full-page"


class OCRResponse(BaseModel):
    """
    Schema for the final OCR response.
    """

    filename: str
    text: str
    confidence: Optional[float] = None
    iterations: Optional[list[OCRIteration]] = None
    processing_time: float
    s3_key: Optional[str] = None
    reconstruction: Optional[dict] = None
    layout_analysis: Optional[dict] = None
    method: Optional[str] = None
    request_id: Optional[str] = None
    document_type: Optional[str] = None
    type_confidence: Optional[float] = None
    card_analysis: Optional[dict] = None


class DocumentField(BaseModel):
    """
    Schema for a single extracted field from a personal document.
    """

    name: str
    value: str
    raw_ocr: Optional[str] = None
    confidence_level: Literal["high", "medium", "low"]


class DocumentAnalytics(BaseModel):
    """
    Analytics block attached to each document OCR response.

    Provides pixel-level quality metrics, decision readiness scoring, and
    remediation hints to guide downstream consumers.
    """

    pixel_coverage_ratio: Optional[float] = None
    readability_index: Optional[float] = None
    decision_readiness: Optional[dict[str, Any]] = None
    iteration_convergence: Optional[float] = None
    pixel_rescue_applied: bool = False
    quality_band: Optional[str] = None
    requires_manual_review: bool = False
    remediation_hints: list[str] = []
    field_completeness_ratio: Optional[float] = None
    fields_extracted_count: int = 0
    fields_expected_count: int = 0


class DocumentResponse(BaseModel):
    """
    Schema for structured personal document OCR response.
    Includes document classification, extracted fields, warnings, and metadata.
    """

    filename: str
    document_type: str
    type_confidence: float
    plain_text: str
    fields: list[DocumentField]
    warnings: list[str]
    metadata: dict[str, Any]
    processing_time: float
    request_id: Optional[str] = None
    s3_key: Optional[str] = None
    analytics: Optional[DocumentAnalytics] = None


class HealthResponse(BaseModel):
    """
    Schema for the health check response.
    """

    status: str
    timestamp: float
    components: Optional[dict] = None


class ReconStatusResponse(BaseModel):
    """
    Schema for the reconstruction status response.
    """

    reconstruction_enabled: bool
    package_installed: bool
    package_version: Optional[str] = None


class PresignRequest(BaseModel):
    """Request to create a presigned S3 POST for direct uploads."""

    key: str
    content_type: str = "application/octet-stream"
    expires_in: int = 3600


class PresignResponse(BaseModel):
    """Response containing the S3 POST URL and form fields."""

    url: str
    fields: dict


class ErrorResponse(BaseModel):
    """
    Standardized error response schema.
    """

    phase: str
    message: str
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    filename: Optional[str] = None

