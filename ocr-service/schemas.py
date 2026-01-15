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

    model_config = {
        "json_schema_extra": {
            "example": {
                "iteration": 1,
                "text_length": 150,
                "confidence": 0.85,
                "preview_text": "Invoice #12345...",
                "method": "full-page",
            }
        }
    }


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
    request_id: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "filename": "document.png",
                "text": "Full extracted text from the document.",
                "confidence": 0.92,
                "iterations": [
                    {
                        "iteration": 1,
                        "text_length": 150,
                        "confidence": 0.85,
                        "preview_text": "Invoice #12345...",
                        "method": "full-page",
                    }
                ],
                "processing_time": 0.45,
                "s3_key": "processed/uuid-document.png",
                "request_id": "rid-12345",
            }
        }
    }


class HealthResponse(BaseModel):
    """
    Schema for the health check response.
    """

    status: str
    timestamp: float
    environment: Optional[str] = None
    services: Optional[dict[str, str]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "timestamp": 1705350000.0,
                "environment": "production",
                "services": {
                    "s3": "healthy",
                    "supabase": "healthy",
                    "openai": "configured",
                    "gemini": "configured",
                },
            }
        }
    }


class ReconStatusResponse(BaseModel):
    """
    Schema for the reconstruction status response.
    """

    reconstruction_enabled: bool
    package_installed: bool
    package_version: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "reconstruction_enabled": True,
                "package_installed": True,
                "package_version": "1.0.0",
            }
        }
    }


class PresignRequest(BaseModel):
    """Request to create a presigned S3 POST for direct uploads."""

    key: str
    content_type: Optional[str] = "application/octet-stream"
    expires_in: Optional[int] = 3600

    model_config = {
        "json_schema_extra": {
            "example": {
                "key": "uploads/document.png",
                "content_type": "image/png",
                "expires_in": 3600,
            }
        }
    }


class PresignResponse(BaseModel):
    """Response containing the S3 POST URL and form fields."""

    url: str
    fields: dict

    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "https://bucket.s3.amazonaws.com/",
                "fields": {
                    "key": "uploads/document.png",
                    "AWSAccessKeyId": "...",
                    "policy": "...",
                    "signature": "...",
                },
            }
        }
    }
