"""
PDF-to-image converter for the OCR pipeline.

Provides utilities to detect PDF bytes and convert PDF pages to BGR numpy
arrays so they can flow through the same pixel pipeline (upscale, enhance,
reconstruct, OCR) as regular images.

Strategy:
  1. Try ``pdf2image`` (requires poppler) for high-quality rasterisation.
  2. Fall back to ``Pillow`` for simpler PDFs when poppler is unavailable.
"""

from __future__ import annotations

import importlib
import io
import logging
from typing import Any

import numpy as np

__all__ = ["is_pdf", "pdf_pages_to_images"]

logger = logging.getLogger("ocr-service.pdf-converter")

_PDF_MAGIC = b"%PDF-"

_DEFAULT_MAX_PAGES = 20
_DEFAULT_DPI = 300


def is_pdf(data: bytes) -> bool:
    """Return True if *data* starts with the PDF magic bytes ``%PDF-``."""
    return data[:5] == _PDF_MAGIC


def pdf_pages_to_images(
    pdf_bytes: bytes,
    dpi: int = _DEFAULT_DPI,
    max_pages: int = _DEFAULT_MAX_PAGES,
) -> list[np.ndarray]:
    """
    Convert the pages of a PDF to BGR numpy arrays.

    Args:
        pdf_bytes: Raw PDF file bytes.
        dpi: Rasterisation resolution (default 300 dpi).
        max_pages: Maximum number of pages to convert (default 20).

    Returns:
        A list of BGR ``np.ndarray`` images, one per page (up to *max_pages*).
        Returns an empty list when conversion fails entirely.

    The function tries ``pdf2image`` (poppler) first for best quality, then
    falls back to ``Pillow`` for simpler PDFs.
    """
    pages = _try_pdf2image(pdf_bytes, dpi, max_pages)
    if pages is not None:
        return pages
    pages = _try_pillow(pdf_bytes, max_pages)
    if pages is not None:
        return pages
    logger.warning("PDF conversion failed: no suitable backend available")
    return []


def _try_pdf2image(
    pdf_bytes: bytes,
    dpi: int,
    max_pages: int,
) -> list[np.ndarray] | None:
    """Attempt conversion via pdf2image (requires poppler)."""
    try:
        pdf2image = importlib.import_module("pdf2image")
    except ImportError:
        return None

    try:
        pil_images = pdf2image.convert_from_bytes(
            pdf_bytes,
            dpi=dpi,
            first_page=1,
            last_page=max_pages,
        )
        return [_pil_to_bgr(img) for img in pil_images]
    except Exception:
        logger.debug("pdf2image conversion failed", exc_info=True)
        return None


def _try_pillow(
    pdf_bytes: bytes,
    max_pages: int,
) -> list[np.ndarray] | None:
    """Attempt conversion via Pillow (works for simple single-image PDFs)."""
    try:
        image_module = importlib.import_module("PIL.Image")
    except ImportError:
        return None

    try:
        results: list[np.ndarray] = []
        with image_module.open(io.BytesIO(pdf_bytes)) as img:
            for page_idx in range(max_pages):
                try:
                    img.seek(page_idx)
                except EOFError:
                    break
                results.append(_pil_to_bgr(img.copy()))
        return results or None
    except Exception:
        logger.debug("Pillow PDF conversion failed", exc_info=True)
        return None


def _pil_to_bgr(pil_image: Any) -> np.ndarray:
    """Convert a PIL Image to an OpenCV-compatible BGR numpy array."""
    cv2 = importlib.import_module("cv2")

    rgb = pil_image.convert("RGB")
    arr = np.array(rgb, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
