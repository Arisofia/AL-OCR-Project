"""
Pydantic schemas for the OCR service API.
"""

from typing import List, Optional

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
    iterations: Optional[List[OCRIteration]] = None
    processing_time: float
    s3_key: Optional[str] = None
    reconstruction: Optional[dict] = None
    layout_analysis: Optional[dict] = None
    method: Optional[str] = None


class HealthResponse(BaseModel):
    """
    Schema for the health check response.
    """

    status: str
    timestamp: float


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
    content_type: Optional[str] = "application/octet-stream"
    expires_in: Optional[int] = 3600


class PresignResponse(BaseModel):
    """Response containing the S3 POST URL and form fields."""

    url: str
    fields: dict
