"""Tests for ocr_service.modules.pdf_converter."""

from ocr_service.modules.pdf_converter import is_pdf


def test_is_pdf_true():
    assert is_pdf(b"%PDF-1.4 some content here") is True


def test_is_pdf_false_png():
    # PNG magic bytes: \x89PNG
    assert is_pdf(b"\x89PNG\r\n\x1a\n") is False


def test_is_pdf_false_empty():
    assert is_pdf(b"") is False


def test_is_pdf_false_jpeg():
    # JPEG magic bytes: \xff\xd8\xff
    assert is_pdf(b"\xff\xd8\xff\xe0") is False


def test_is_pdf_false_short():
    assert is_pdf(b"%PDF") is False  # only 4 bytes, magic is 5
