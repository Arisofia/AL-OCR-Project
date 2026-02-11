"""
Custom exceptions for the OCR service.
"""


class OCRPipelineError(Exception):
    """Custom exception for OCR pipeline errors."""

    def __init__(
        self,
        phase: str,
        message: str,
        status_code: int = 500,
        correlation_id: str | None = None,
        trace_id: str | None = None,
        filename: str | None = None,
    ):
        self.phase = phase
        self.message = message
        self.status_code = status_code
        self.correlation_id = correlation_id
        self.trace_id = trace_id
        self.filename = filename
        super().__init__(self.message)
