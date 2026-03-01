"""Tests for ocr_service.modules.decision_readiness."""

import pytest

from ocr_service.modules.decision_readiness import (
    MANDATORY_FIELDS,
    compute_decision_readiness,
)
from ocr_service.modules.personal_doc_extractor import ExtractedField


def _field(name: str, confidence: str = "high") -> ExtractedField:
    return ExtractedField(name=name, value="value", confidence_level=confidence)


# ---------------------------------------------------------------------------
# Core scoring tests
# ---------------------------------------------------------------------------


def test_passport_all_fields_high_confidence():
    fields = [_field(f) for f in MANDATORY_FIELDS["passport"]]
    result = compute_decision_readiness("passport", fields, type_confidence=0.90)
    assert result["ready"] is True
    assert result["score"] >= 0.70
    assert result["missing_mandatory"] == []


def test_passport_missing_fields():
    # Only full_name and date_of_birth present; document_number + expiry_date missing
    # With low type_confidence:
    # 0.50*0.50 + 1.0*0.30 + 0.40*0.20 = 0.25+0.30+0.08 = 0.63 < 0.70
    fields = [_field("full_name"), _field("date_of_birth")]
    result = compute_decision_readiness("passport", fields, type_confidence=0.40)
    assert result["ready"] is False
    assert "document_number" in result["missing_mandatory"]
    assert "expiry_date" in result["missing_mandatory"]


def test_low_confidence_fields():
    """All mandatory fields present but all low confidence - score < 0.70."""
    fields = [_field(f, "low") for f in MANDATORY_FIELDS["passport"]]
    result = compute_decision_readiness("passport", fields, type_confidence=0.50)
    # presence=1.0*0.50 + avg_conf=0.3*0.30 + type=0.5*0.20 = 0.50+0.09+0.10 = 0.69
    assert result["score"] < 0.70
    assert result["ready"] is False


def test_unknown_document_type():
    result = compute_decision_readiness("alien_form", [], type_confidence=0.50)
    assert result["ready"] is False
    assert "unknown document type" in result["recommendation"].lower()


def test_bank_card_readiness():
    fields = [_field("card_number"), _field("expiry_date")]
    result = compute_decision_readiness("bank_card", fields, type_confidence=0.85)
    assert result["ready"] is True
    assert result["missing_mandatory"] == []


@pytest.mark.parametrize("doc_type", list(MANDATORY_FIELDS.keys()))
def test_all_doc_types_have_mandatory_fields(doc_type):
    """Every known doc type must define at least one mandatory field."""
    assert len(MANDATORY_FIELDS[doc_type]) >= 1


def test_recommendation_is_string():
    fields = [_field("total_amount")]
    result = compute_decision_readiness("invoice", fields, type_confidence=0.70)
    assert isinstance(result["recommendation"], str)


def test_score_bounded_between_0_and_1():
    fields = [_field(f) for f in MANDATORY_FIELDS["passport"]]
    result = compute_decision_readiness("passport", fields, type_confidence=1.0)
    assert 0.0 <= result["score"] <= 1.0
