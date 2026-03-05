"""Exception-path tests for learning engine pattern loading."""

import logging
import os
import tempfile

import pytest

from ocr_service.modules.learning_engine import LearningEngine


def test_load_patterns_logs_on_invalid_json(caplog):
    caplog.set_level(logging.ERROR)
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write("{bad json")
        tmp_path = tmp.name

    engine = LearningEngine()
    engine.storage_path = tmp_path

    res = engine._load_patterns()
    assert res == []
    assert any(
        "Failed to load or parse local learning patterns" in r.message
        for r in caplog.records
    )

    os.unlink(tmp_path)
