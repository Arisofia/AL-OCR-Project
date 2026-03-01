"""
TestClient-based endpoint tests for POST /ocr/documents.

Covers:
- Required fields in DocumentResponse (response contract)
- PAN (card_number) masked to last 4 digits; raw_ocr == "[REDACTED]"
- CVV completely absent from the response
- Requests without a valid API key rejected with 403
"""

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from ocr_service.config import get_settings
from ocr_service.main import app
from ocr_service.routers.deps import get_ocr_processor

# ---------------------------------------------------------------------------
# Synthetic OCR processor results
# ---------------------------------------------------------------------------

_PASSPORT_RESULT = {
    "filename": "passport.jpg",
    "text": (
        "PASSPORT\n"
        "Surname: SMITH\n"
        "Given Names: JOHN\n"
        "Nationality: GBR\n"
        "Date of Birth: 15/03/1985\n"
        "Passport No: AB123456\n"
        "Date of Expiry: 25/09/2030\n"
    ),
    "document_type": "passport",
    "type_confidence": 0.80,
    "confidence": 0.95,
    "processing_time": 0.5,
    "method": "gemini",
}

_BANK_CARD_RESULT = {
    "filename": "card.jpg",
    "text": "VISA\n4111 1111 1111 1111\nJOHN SMITH\nEXP 12/26\nCVV 123\n",
    "document_type": "bank_card",
    "type_confidence": 0.85,
    "confidence": 0.92,
    "processing_time": 0.3,
    "method": "gemini",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth_headers():
    settings = get_settings()
    return {str(settings.api_key_header_name): str(settings.ocr_api_key)}


def _fake_upload(filename="test.jpg"):
    return {"file": (filename, io.BytesIO(b"fake-image-bytes"), "image/jpeg")}


def _make_mock_processor(result: dict) -> MagicMock:
    mock = MagicMock()
    mock.process_file = AsyncMock(return_value=result)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_document_endpoint_response_contract():
    """All DocumentResponse required fields must be present in the response."""
    app.dependency_overrides[get_ocr_processor] = lambda: _make_mock_processor(
        _PASSPORT_RESULT
    )
    try:
        client = TestClient(app)
        response = client.post(
            "/ocr/documents",
            files=_fake_upload(),
            headers=_auth_headers(),
        )
        assert response.status_code == 200, response.text
        data = response.json()

        required_keys = (
            "filename",
            "document_type",
            "type_confidence",
            "plain_text",
            "fields",
            "warnings",
            "metadata",
            "processing_time",
            # Optional fields (request_id, s3_key) are not checked here
        )
        for key in required_keys:
            assert key in data, f"Missing required key in DocumentResponse: {key!r}"

        assert data["document_type"] == "passport"
        assert isinstance(data["type_confidence"], float)
        assert 0.0 <= data["type_confidence"] <= 1.0
        assert isinstance(data["fields"], list)
        assert isinstance(data["warnings"], list)
        assert isinstance(data["metadata"], dict)
        assert "language_guess" in data["metadata"]
    finally:
        app.dependency_overrides.pop(get_ocr_processor, None)


def test_document_endpoint_pan_masked():
    """PAN must be masked to last 4 digits; raw_ocr must be [REDACTED]."""
    app.dependency_overrides[get_ocr_processor] = lambda: _make_mock_processor(
        _BANK_CARD_RESULT
    )
    try:
        client = TestClient(app)
        response = client.post(
            "/ocr/documents",
            files=_fake_upload(),
            headers=_auth_headers(),
        )
        assert response.status_code == 200, response.text
        data = response.json()

        card_field = next(
            (f for f in data["fields"] if f["name"] == "card_number"), None
        )
        assert card_field is not None, "card_number field should be present"
        # Last 4 digits visible
        assert "1111" in card_field["value"]
        # Full PAN must not appear in value
        assert "4111 1111 1111 1111" not in card_field["value"]
        # Masking format must show leading asterisks followed by last 4 digits
        assert card_field["value"].endswith("1111"), card_field["value"]
        assert "*" in card_field["value"]
        # raw_ocr must be redacted
        assert card_field["raw_ocr"] == "[REDACTED]"
    finally:
        app.dependency_overrides.pop(get_ocr_processor, None)


def test_document_endpoint_cvv_omitted():
    """CVV must be completely absent from the /ocr/documents response."""
    app.dependency_overrides[get_ocr_processor] = lambda: _make_mock_processor(
        _BANK_CARD_RESULT
    )
    try:
        client = TestClient(app)
        response = client.post(
            "/ocr/documents",
            files=_fake_upload(),
            headers=_auth_headers(),
        )
        assert response.status_code == 200, response.text
        data = response.json()

        cvv_fields = [
            f for f in data["fields"] if f["name"] in {"cvv", "cvc", "cvv2", "cvc2"}
        ]
        assert cvv_fields == [], "CVV must not appear in /ocr/documents response"
    finally:
        app.dependency_overrides.pop(get_ocr_processor, None)


def test_document_endpoint_requires_auth():
    """Requests without a valid API key must be rejected with 403."""
    app.dependency_overrides[get_ocr_processor] = lambda: _make_mock_processor(
        _PASSPORT_RESULT
    )
    try:
        client = TestClient(app)
        response = client.post("/ocr/documents", files=_fake_upload())
        assert response.status_code == 403
    finally:
        app.dependency_overrides.pop(get_ocr_processor, None)


def test_document_endpoint_luhn_note_in_warnings():
    """A valid Luhn PAN must produce a Luhn advisory note in the warnings array."""
    app.dependency_overrides[get_ocr_processor] = lambda: _make_mock_processor(
        _BANK_CARD_RESULT
    )
    try:
        client = TestClient(app)
        response = client.post(
            "/ocr/documents",
            files=_fake_upload(),
            headers=_auth_headers(),
        )
        assert response.status_code == 200, response.text
        data = response.json()
        warnings = data.get("warnings", [])
        # 4111111111111111 is Luhn-valid, so a positive Luhn note must appear
        assert any("Luhn" in w for w in warnings), (
            f"Expected a Luhn advisory note in warnings; got: {warnings}"
        )
    finally:
        app.dependency_overrides.pop(get_ocr_processor, None)


def test_document_endpoint_expiry_format_note_in_warnings():
    """Valid MM/YY expiry on a bank card must produce an expiry format note."""
    app.dependency_overrides[get_ocr_processor] = lambda: _make_mock_processor(
        _BANK_CARD_RESULT
    )
    try:
        client = TestClient(app)
        response = client.post(
            "/ocr/documents",
            files=_fake_upload(),
            headers=_auth_headers(),
        )
        assert response.status_code == 200, response.text
        data = response.json()
        warnings = data.get("warnings", [])
        # EXP 12/26 → "Expiry date format valid (MM/YY)"
        assert any("format valid" in w for w in warnings), (
            f"Expected expiry format note in warnings; got: {warnings}"
        )
    finally:
        app.dependency_overrides.pop(get_ocr_processor, None)
