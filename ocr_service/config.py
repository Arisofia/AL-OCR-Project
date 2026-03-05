"""
Configuration module for the OCR service.

This module defines the settings schema using Pydantic BaseSettings,
supporting environment variable overrides and LRU caching for performance.
"""

from functools import lru_cache
from typing import Any, Literal, Optional

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
        redis_host: Redis host address.
        redis_port: Redis port number.
        redis_db: Redis database index.
    """

    app_name: str = "AL Financial OCR Project"
    app_description: str = "Professional Iterative OCR & Pixel Reconstruction Service"
    version: str = "1.2.0"

    ocr_api_key: str = Field(
        default="dev-secret-key-change-in-production",
        description="Secret key for OCR authentication"
    )
    api_key_header_name: str = "X-API-KEY"

    s3_bucket_name: Optional[str] = None
    output_prefix: str = "textract_outputs/"

    aws_max_retries: int = 3
    aws_region: str = "us-east-1"

    enable_reconstruction: bool = False
    ocr_iterations: int = 3
    ocr_card_iterations: int = 1
    ocr_card_pass_limit: int = 2
    ocr_card_timeout_seconds: float = 8.0
    ocr_strategy_profile: Literal[
        "deterministic", "layout_aware", "hybrid"
    ] = "hybrid"
    ocr_doc_type_strategy_overrides: dict[
        str, Literal["deterministic", "layout_aware", "hybrid"]
    ] = Field(default_factory=dict)
    enable_bin_lookup: bool = False

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_startup_check: bool = True
    ocr_idempotency_ttl_seconds: int = 3600

    environment: str = "development"
    allowed_origins: list[str] = ["*"]

    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    hugging_face_hub_token: Optional[str] = None
    perplexity_api_key: Optional[str] = None

    supabase_url: Optional[str] = None
    supabase_anon_key: Optional[str] = None
    supabase_service_role: Optional[str] = None
    use_local_fallback: bool = True
    local_data_path: str = "data/learning_patterns.json"

    sentry_dsn: Optional[str] = None
    azure_application_insights_connection_string: Optional[str] = None

    redis_idempotency_ttl: int = 3600

    drift_report_path: str = "reports/drift_report.html"
    reference_baseline_path: str = "data/reference_baseline.csv"
    al_cycle_samples: int = 50
    al_n_clusters: int = 5

    dataset_upload_key: Optional[str] = None

    @field_validator("allowed_origins", mode="after")
    @classmethod
    def validate_origins(cls, v: list[str], info) -> list[str]:
        """Ensures that wildcards are not used for CORS in production."""
        if info.data.get("environment") == "production" and "*" in v:
            raise ValueError("Wildcard CORS origins are not allowed in production.")
        return v

    @field_validator("ocr_strategy_profile", mode="before")
    @classmethod
    def normalize_ocr_strategy_profile(cls, value: str) -> str:
        """Normalize OCR strategy profile input before Literal validation."""
        return (value or "hybrid").strip().lower()

    @field_validator("ocr_doc_type_strategy_overrides", mode="before")
    @classmethod
    def normalize_doc_type_strategy_overrides(
        cls, value: Any
    ) -> dict[str, str]:
        """Normalize document-type strategy override keys and values."""
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            return value
        return {
            (str(key) or "").strip().lower(): (str(strategy) or "hybrid")
            .strip()
            .lower()
            for key, strategy in value.items()
        }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


try:
    from opentelemetry import trace

    TRACER = trace.get_tracer(__name__)
except ImportError:
    TRACER = None


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached instance of the application settings.
    """
    if TRACER:
        with TRACER.start_as_current_span("load_settings"):
            return Settings()
    return Settings()
