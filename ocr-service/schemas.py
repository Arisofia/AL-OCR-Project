from pydantic import BaseModel
from typing import List, Optional

class OCRIteration(BaseModel):
    iteration: int
    text_length: int
    preview_text: str

class OCRResponse(BaseModel):
    filename: str
    text: str
    iterations: List[OCRIteration]
    processing_time: float
    s3_key: Optional[str] = None
    reconstruction: Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: float

class ReconStatusResponse(BaseModel):
    reconstruction_enabled: bool
    package_installed: bool
    package_version: Optional[str]
