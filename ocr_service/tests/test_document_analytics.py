"""Tests for DocumentAnalytics schema and analytics wiring."""

import pytest

from ocr_service.modules.decision_readiness import quality_band as _quality_band
from ocr_service.schemas import DocumentAnalytics, DocumentResponse

# ---------------------------------------------------------------------------
# Schema unit tests
# ---------------------------------------------------------------------------


def test_analytics_field_present_in_document_response():
    """DocumentResponse must have an optional analytics field."""
    resp = DocumentResponse(
        filename="test.jpg",
        document_type="passport",
        type_confidence=0.90,
        plain_text="PASSPORT",
        fields=[],
        warnings=[],
        metadata={},
        processing_time=1.0,
        analytics=None,
    )
    assert hasattr(resp, "analytics")
    assert resp.analytics is None


def test_document_response_with_analytics():
    analytics = DocumentAnalytics(
        quality_band="excellent",
        fields_extracted_count=4,
        fields_expected_count=4,
        field_completeness_ratio=1.0,
        requires_manual_review=False,
    )
    resp = DocumentResponse(
        filename="test.jpg",
        document_type="passport",
        type_confidence=0.90,
        plain_text="PASSPORT",
        fields=[],
        warnings=[],
        metadata={},
        processing_time=1.0,
        analytics=analytics,
    )
    assert resp.analytics is not None
    assert resp.analytics.quality_band == "excellent"


def test_quality_band_excellent():
    a = DocumentAnalytics(quality_band="excellent")
    assert a.quality_band == "excellent"


def test_quality_band_good():
    a = DocumentAnalytics(quality_band="good")
    assert a.quality_band == "good"


def test_quality_band_fair():
    a = DocumentAnalytics(quality_band="fair")
    assert a.quality_band == "fair"


def test_quality_band_poor():
    a = DocumentAnalytics(quality_band="poor")
    assert a.quality_band == "poor"


def test_remediation_hints_defaults_empty():
    a = DocumentAnalytics()
    assert a.remediation_hints == []


def test_remediation_hints_for_missing_fields():
    hints = [
        "Mandatory field 'document_number' could not be extracted; verify manually."
    ]
    a = DocumentAnalytics(remediation_hints=hints)
    assert "document_number" in a.remediation_hints[0]


def test_analytics_defaults():
    a = DocumentAnalytics()
    assert a.pixel_rescue_applied is False
    assert a.requires_manual_review is False
    assert a.fields_extracted_count == 0
    assert a.fields_expected_count == 0


# ---------------------------------------------------------------------------
# Quality band logic (from decision_readiness module)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("confidence,expected", [
    (0.90, "excellent"),
    (0.85, "excellent"),
    (0.70, "good"),
    (0.65, "good"),
    (0.50, "fair"),
    (0.40, "fair"),
    (0.39, "poor"),
    (0.00, "poor"),
])
def test_quality_band_thresholds(confidence, expected):
    assert _quality_band(confidence) == expected
