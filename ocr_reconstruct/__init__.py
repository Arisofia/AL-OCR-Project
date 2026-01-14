"""ocr_reconstruct package — exposes the reconstruction pipeline
as a standard importable package."""

from .modules.pipeline import process_bytes

__all__ = ["process_bytes"]
__version__ = "0.1.0"
