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
    max_image_size_mb: int = 10
    default_doc_type: str = "generic"
    enable_reconstruction: bool = False
    confidence_threshold: float = 0.5
