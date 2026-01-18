"""
Configuration module for the OCR service.

This module defines the settings schema using Pydantic BaseSettings,
supporting environment variable overrides and LRU caching for performance.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings for the OCR service.

    Attributes:
        app_name: Name of the application.
        app_description: Brief description of the OCR service.
        version: Application version.
        ocr_api_key: Secret key for OCR authentication.
        api_key_header_name: Name of the HTTP header for API key.
        s3_bucket_name: Optional S3 bucket name for storage.
        output_prefix: Prefix for Textract output files in S3.
        aws_max_retries: Maximum number of retries for AWS operations.
        aws_region: AWS region for services.
        enable_reconstruction: Flag to enable pixel reconstruction.
        ocr_iterations: Number of OCR iterations to perform.
    """

    app_name: str = "AL Financial OCR Project"
    app_description: str = "Professional Iterative OCR & Pixel Reconstruction Service"
    version: str = "1.2.0"

    ocr_api_key: str = Field(
        "default_secret_key", description="Secret key for OCR authentication"
    )
    api_key_header_name: str = "X-API-KEY"

    s3_bucket_name: Optional[str] = None
    output_prefix: str = "textract_outputs/"

    aws_max_retries: int = 3
    aws_region: str = "us-east-1"

    enable_reconstruction: bool = False
    ocr_iterations: int = 3

    # Security and Environment
    environment: str = "development"
    allowed_origins: list[str] = ["*"]

    # External AI Services
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    hugging_face_hub_token: Optional[str] = None
    perplexity_api_key: Optional[str] = None

    # Database & Storage
    supabase_url: Optional[str] = None
    supabase_anon_key: Optional[str] = None
    supabase_service_role: Optional[str] = None
    use_local_fallback: bool = True
    local_data_path: str = "data/learning_patterns.json"

    # Monitoring & Logging
    sentry_dsn: Optional[str] = None
    azure_application_insights_connection_string: Optional[str] = None

    # Active Learning & Drift
    drift_report_path: str = "reports/drift_report.html"
    reference_baseline_path: str = "data/reference_baseline.csv"
    al_cycle_samples: int = 50
    al_n_clusters: int = 5

    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, v: list[str], info) -> list[str]:
        """Ensures that wildcards are not used for CORS in production."""
        if info.data.get("environment") == "production" and "*" in v:
            raise ValueError("Wildcard CORS origins are not allowed in production.")
        return v

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached instance of the application settings.
    """
    return Settings()  # type: ignore[call-arg]
