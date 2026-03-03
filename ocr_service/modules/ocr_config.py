"""
Configuration models for OCR engines.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TesseractConfig(BaseModel):
    """Configuration for Tesseract OCR engine."""

    oem: int = Field(default=3, ge=0, le=3)
    psm: int = Field(default=6, ge=0, le=13)
    lang: str = Field(default="spa+eng", min_length=2, max_length=32)

    @property
    def flags(self) -> str:
        """Return CLI flags for Tesseract based on configured OCR options."""
        return f"--oem {self.oem} --psm {self.psm} -l {self.lang}"


class EngineConfig(BaseModel):
    """General engine configuration."""

    max_iterations: int = Field(default=3, ge=1, le=10)
    max_iterations_card: int = Field(default=3, ge=1, le=10)
    max_image_size_mb: int = Field(default=10, ge=1, le=50)
    default_doc_type: str = "generic"
    enable_reconstruction: bool = False
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_upscale_factor: float = Field(default=2.0, ge=1.0, le=4.0)
    max_long_side_px: int = Field(default=3000, ge=512, le=12000)
    enable_bin_lookup: bool = False
    card_ocr_pass_limit: int = Field(default=2, ge=1, le=10)
    card_ocr_timeout_seconds: float = Field(default=8.0, gt=0.0, le=60.0)
    ocr_strategy_profile: Literal[
        "deterministic", "layout_aware", "hybrid"
    ] = "hybrid"

    @field_validator("ocr_strategy_profile", mode="before")
    @classmethod
    def normalize_strategy_profile(cls, value: str) -> str:
        """Normalize OCR strategy profile input before Literal validation."""
        return (value or "hybrid").strip().lower()

    @property
    def normalized_strategy_profile(self) -> str:
        """Return a normalized OCR strategy profile with a safe default."""
        profile = (self.ocr_strategy_profile or "hybrid").strip().lower()
        if profile in {"deterministic", "layout_aware", "hybrid"}:
            return profile
        return "hybrid"

    def allows_vision_quality_fallback(self) -> bool:
        """Return True when non-deterministic fallback routes are allowed."""
        return self.normalized_strategy_profile != "deterministic"

    def prefers_layout_regions(self) -> bool:
        """Return True when layout-aware OCR region extraction is preferred."""
        return self.normalized_strategy_profile == "layout_aware"

    def effective_use_reconstruction(self, requested: bool) -> bool:
        """Resolve whether reconstruction should run for the current strategy."""
        return self.prefers_layout_regions() or requested
