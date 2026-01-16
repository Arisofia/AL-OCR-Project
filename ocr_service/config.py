"""
Configuration module for the OCR service.

This module defines the settings schema using Pydantic BaseSettings,
supporting environment variable overrides and LRU caching for performance.
"""

import logging
from functools import lru_cache
from typing import Optional
from pydantic import Field, ValidationError, model_validator
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

    ocr_api_key: Optional[str] = Field(
        None,
        description="Secret key for OCR authentication (set via env or Secrets)",
        repr=False,
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

    openai_api_key: Optional[str] = Field(None, repr=False)
    gemini_api_key: Optional[str] = Field(None, repr=False)
    hugging_face_hub_token: Optional[str] = Field(None, repr=False)
    perplexity_api_key: Optional[str] = Field(None, repr=False)

    # Database & Storage (Supabase)

    supabase_url: Optional[str] = None
    supabase_anon_key: Optional[str] = Field(None, repr=False)
    supabase_service_role: Optional[str] = Field(None, repr=False)

    # Monitoring & Logging

    sentry_dsn: Optional[str] = Field(None, repr=False)
    azure_application_insights_connection_string: Optional[str] = Field(
        None, repr=False
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @model_validator(mode="after")
    def validate_required_secrets(self) -> "Settings":
        missing = []
        if not self.ocr_api_key:
            missing.append("ocr_api_key")
        if self.environment == "production":
            # In production, require Sentry and at least one AI key
            if not self.sentry_dsn:
                missing.append("sentry_dsn")
            if not (
                self.openai_api_key
                or self.gemini_api_key
                or self.hugging_face_hub_token
                or self.perplexity_api_key
            ):
                missing.append("AI provider key (openai/gemini/huggingface/perplexity)")
        if missing:
            logging.error("Missing required secrets: %s", ", ".join(missing))
            raise ValueError(f"Missing required secrets: {', '.join(missing)}")
        return self


def _log_settings_load() -> None:
    logging.info("Loading application settings from environment and .env file.")


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached instance of the application settings.
    """
    _log_settings_load()
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as e:
        logging.error("Settings validation error: %s", e)
        raise
