"""Unit tests for OCR engine configuration validation."""

import pytest
from pydantic import ValidationError

from ocr_service.modules.ocr_config import EngineConfig, TesseractConfig


def test_tesseract_config_defaults_are_valid():
    """Default Tesseract configuration should validate successfully."""
    config = TesseractConfig()
    assert config.oem == 3
    assert config.psm == 6
    assert config.lang == "spa+eng"


def test_tesseract_config_rejects_invalid_psm():
    """PSM values outside valid Tesseract range should fail validation."""
    with pytest.raises(ValidationError):
        TesseractConfig(psm=99)


def test_engine_config_normalizes_strategy_profile_case():
    """Strategy profile should normalize mixed-case input to canonical value."""
    config = EngineConfig.model_validate({"ocr_strategy_profile": "Hybrid"})
    assert config.ocr_strategy_profile == "hybrid"


def test_engine_config_rejects_invalid_strategy_profile():
    """Unsupported OCR strategy profile should raise validation error."""
    with pytest.raises(ValidationError):
        EngineConfig.model_validate({"ocr_strategy_profile": "experimental"})


def test_engine_config_rejects_invalid_threshold():
    """Confidence threshold outside [0,1] should fail validation."""
    with pytest.raises(ValidationError):
        EngineConfig(confidence_threshold=1.5)


def test_engine_config_rejects_invalid_card_timeout():
    """Card OCR timeout must be positive and within allowed range."""
    with pytest.raises(ValidationError):
        EngineConfig(card_ocr_timeout_seconds=0)


def test_engine_config_uses_layout_aware_for_bank_statement_by_default():
    """Bank statements should default to layout-aware strategy for better fidelity."""
    config = EngineConfig(ocr_strategy_profile="deterministic")
    assert config.strategy_profile_for_doc_type("bank_statement") == "layout_aware"
    assert config.effective_use_reconstruction(False, "bank_statement") is True


def test_engine_config_respects_doc_type_strategy_override():
    """Explicit document-type override should take precedence over defaults."""
    config = EngineConfig(
        ocr_strategy_profile="layout_aware",
        doc_type_strategy_overrides={"receipt": "deterministic"},
    )
    assert config.strategy_profile_for_doc_type("receipt") == "deterministic"
    assert config.allows_vision_quality_fallback("receipt") is False
