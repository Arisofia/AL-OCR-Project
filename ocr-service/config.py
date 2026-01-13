from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
from typing import Optional
import os

class Settings(BaseSettings):
    app_name: str = "AL Financial OCR Project"
    app_description: str = "Professional Iterative OCR & Pixel Reconstruction Service"
    version: str = "1.2.0"
    
    ocr_api_key: str = Field("default_secret_key")
    api_key_header_name: str = "X-API-KEY"
    
    s3_bucket_name: Optional[str] = None
    output_prefix: str = "textract_outputs/"
    
    aws_max_retries: int = 3
    aws_region: str = "us-east-1"
    
    enable_reconstruction: bool = False
    ocr_iterations: int = 3

    # External AI Services
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    hugging_face_hub_token: Optional[str] = None
    perplexity_api_key: Optional[str] = None

    # Database & Storage (Supabase)
    supabase_url: Optional[str] = None
    supabase_anon_key: Optional[str] = None
    supabase_service_role: Optional[str] = None

    # Monitoring & Logging
    sentry_dsn: Optional[str] = None
    azure_application_insights_connection_string: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache()
def get_settings():
    return Settings()
