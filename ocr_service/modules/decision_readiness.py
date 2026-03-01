"""
Decision readiness scoring for extracted document fields.

Computes a numeric readiness score that reflects how suitable a document's
extracted data is for automated downstream decisions (e.g., KYC, lending).

Score formula:
    score = (presence_ratio * 0.50) + (avg_field_confidence * 0.30)
            + (type_confidence * 0.20)

Ready threshold: score >= 0.70
"""

from __future__ import annotations

from typing import Any

__all__ = ["compute_decision_readiness", "quality_band", "MANDATORY_FIELDS"]

# ---------------------------------------------------------------------------
# Mandatory field definitions per document type
# ---------------------------------------------------------------------------

MANDATORY_FIELDS: dict[str, list[str]] = {
    "passport": ["full_name", "document_number", "date_of_birth", "expiry_date"],
    "national_id": ["full_name", "document_number", "date_of_birth"],
    "id_document": ["full_name", "document_number", "date_of_birth"],
    "id_card": ["full_name", "document_number", "date_of_birth"],
    "driver_license": ["full_name", "document_number", "date_of_birth"],
    "bank_card": ["card_number", "expiry_date"],
    "credit_card": ["card_number", "expiry_date"],
    "debit_card": ["card_number", "expiry_date"],
    "bank_statement": ["full_name", "account_number"],
    "utility_bill": ["full_name", "total_amount"],
    "payslip": ["full_name", "salary"],
    "invoice": ["total_amount"],
    "tax_id": ["full_name", "tax_number"],
    "employment_letter": ["full_name", "employer"],
    "residence_permit": ["full_name", "document_number", "expiry_date"],
}

# Confidence level → numeric weight
_CONFIDENCE_WEIGHTS: dict[str, float] = {
    "high": 1.0,
    "medium": 0.7,
    "low": 0.3,
}

_READY_THRESHOLD = 0.70


def quality_band(confidence: float) -> str:
    """Classify OCR/document confidence into a quality band.

    Returns ``"excellent"`` (>=0.85), ``"good"`` (>=0.65),
    ``"fair"`` (>=0.40), or ``"poor"`` (<0.40).
    """
    if confidence >= 0.85:
        return "excellent"
    if confidence >= 0.65:
        return "good"
    if confidence >= 0.40:
        return "fair"
    return "poor"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_decision_readiness(
    document_type: str,
    fields: list[Any],
    type_confidence: float,
) -> dict[str, Any]:
    """
    Compute a decision readiness score for a set of extracted document fields.

    Args:
        document_type: Classified document type (e.g. ``"passport"``).
        fields: List of :class:`ExtractedField`-like objects with ``name``
                and ``confidence_level`` attributes.
        type_confidence: Float in ``[0, 1]`` representing classification
                         confidence.

    Returns:
        A dict with keys:
        - ``score`` (float)       - overall readiness score in ``[0, 1]``
        - ``ready`` (bool)        - True when ``score >= 0.70``
        - ``missing_mandatory``   - list of mandatory fields not present
        - ``recommendation``      - human-readable guidance string
    """
    mandatory = MANDATORY_FIELDS.get(document_type)

    if mandatory is None:
        return {
            "score": 0.0,
            "ready": False,
            "missing_mandatory": [],
            "recommendation": (
                f"Unknown document type '{document_type}'. "
                "Manual review required."
            ),
        }

    extracted: dict[str, str] = {
        f.name: f.confidence_level for f in fields
    }

    # --- presence ratio (mandatory fields present vs total mandatory) --------
    present = [f for f in mandatory if f in extracted]
    missing = [f for f in mandatory if f not in extracted]
    presence_ratio = len(present) / len(mandatory) if mandatory else 1.0

    # --- average confidence of mandatory fields that are present -------------
    confidence_scores = [
        _CONFIDENCE_WEIGHTS.get(extracted[f], 0.3) for f in present
    ]
    avg_field_confidence = (
        sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    )

    # --- composite score -----------------------------------------------------
    score = (
        presence_ratio * 0.50
        + avg_field_confidence * 0.30
        + float(type_confidence) * 0.20
    )
    score = round(min(max(score, 0.0), 1.0), 4)
    ready = score >= _READY_THRESHOLD

    # --- recommendation ------------------------------------------------------
    if ready:
        recommendation = "Document is ready for automated processing."
    elif missing:
        recommendation = (
            f"Missing mandatory fields: {', '.join(missing)}. "
            "Manual review recommended."
        )
    else:
        recommendation = (
            "Low confidence on extracted fields. Manual review recommended."
        )

    return {
        "score": score,
        "ready": ready,
        "missing_mandatory": missing,
        "recommendation": recommendation,
    }
