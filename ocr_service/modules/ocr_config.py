"""
Configuration models for OCR engines.
"""

from pydantic import BaseModel


class TesseractConfig(BaseModel):
    """Configuration for Tesseract OCR engine."""

    oem: int = 3
    psm: int = 6
    lang: str = "spa+eng"

    @property
    def flags(self) -> str:
        return f"--oem {self.oem} --psm {self.psm} -l {self.lang}"


class EngineConfig(BaseModel):
    """General engine configuration."""

    max_iterations: int = 3
    max_iterations_card: int = 3
    max_image_size_mb: int = 10
    default_doc_type: str = "generic"
    enable_reconstruction: bool = False
    confidence_threshold: float = 0.5
    max_upscale_factor: float = 2.0
    max_long_side_px: int = 3000
    enable_bin_lookup: bool = False
    card_ocr_pass_limit: int = 2
    card_ocr_timeout_seconds: float = 8.0
    ocr_strategy_profile: str = "hybrid"

    @property
    def normalized_strategy_profile(self) -> str:
        profile = (self.ocr_strategy_profile or "hybrid").strip().lower()
        if profile in {"deterministic", "layout_aware", "hybrid"}:
            return profile
        return "hybrid"

    def allows_vision_quality_fallback(self) -> bool:
        return self.normalized_strategy_profile != "deterministic"

    def prefers_layout_regions(self) -> bool:
        return self.normalized_strategy_profile == "layout_aware"

    def effective_use_reconstruction(self, requested: bool) -> bool:
        if self.prefers_layout_regions():
            return True
        return requested
