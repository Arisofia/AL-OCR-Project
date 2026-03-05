"""
Drift Detection Service for OCR deployment monitoring.
Uses evidently to compare production data against training reference.
"""

import logging
import os
from importlib import import_module
from typing import Any, Optional

import pandas as pd

from ocr_service.config import get_settings

DATA_DRIFT_PRESET_CLS = None
REPORT_CLS = None
TEST_SUITE_CLS = None
TEST_DRIFTED_COLUMNS_CLS = None

try:
    METRIC_PRESET_MOD = import_module("evidently.metric_preset")
    REPORT_MOD = import_module("evidently.report")
    TEST_SUITE_MOD = import_module("evidently.test_suite")
    TESTS_MOD = import_module("evidently.tests")
except ImportError:
    METRIC_PRESET_MOD = None
    REPORT_MOD = None
    TEST_SUITE_MOD = None
    TESTS_MOD = None

if METRIC_PRESET_MOD is not None:
    DATA_DRIFT_PRESET_CLS = getattr(METRIC_PRESET_MOD, "DataDriftPreset", None)
if REPORT_MOD is not None:
    REPORT_CLS = getattr(REPORT_MOD, "Report", None)
if TEST_SUITE_MOD is not None:
    TEST_SUITE_CLS = getattr(TEST_SUITE_MOD, "TestSuite", None)
if TESTS_MOD is not None:
    TEST_DRIFTED_COLUMNS_CLS = getattr(TESTS_MOD, "TestNumberOfDriftedColumns", None)

logger = logging.getLogger("ocr-service.drift")


def _run_drift_report_and_suite(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    actual_report_path: str,
    report_cls: Any,
    data_drift_preset_cls: Any,
    test_suite_cls: Any,
    test_drifted_columns_cls: Any,
) -> bool:
    """Generate Evidently report + test suite and return whether drift is detected."""
    drift_report = report_cls(metrics=[data_drift_preset_cls()])
    drift_report.run(reference_data=reference_data, current_data=current_data)

    os.makedirs(os.path.dirname(actual_report_path), exist_ok=True)
    drift_report.save_html(actual_report_path)

    data_test = test_suite_cls(tests=[test_drifted_columns_cls()])
    data_test.run(reference_data=reference_data, current_data=current_data)

    summary = data_test.as_dict()["summary"]
    if not summary["all_passed"]:
        logger.warning(
            "DRIFT DETECTED! Significant divergence in input data distribution."
        )
        return True

    logger.info("No significant drift detected.")
    return False


def check_for_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    report_path: Optional[str] = None,
) -> bool:
    """
    Compares 'reference_data' (training set) vs 'current_data' (production inference).
    Focuses on metadata drift and model confidence drift.
    Returns True if significant drift is detected.
    """
    settings = get_settings()
    actual_report_path = report_path or settings.drift_report_path

    if (
        REPORT_CLS is None
        or DATA_DRIFT_PRESET_CLS is None
        or TEST_SUITE_CLS is None
        or TEST_DRIFTED_COLUMNS_CLS is None
    ):
        logger.warning("Evidently is not installed; drift detection skipped.")
        return False

    try:
        return _run_drift_report_and_suite(
            reference_data=reference_data,
            current_data=current_data,
            actual_report_path=actual_report_path,
            report_cls=REPORT_CLS,
            data_drift_preset_cls=DATA_DRIFT_PRESET_CLS,
            test_suite_cls=TEST_SUITE_CLS,
            test_drifted_columns_cls=TEST_DRIFTED_COLUMNS_CLS,
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as e:
        logger.exception("Drift detection failed: %s", e)
        return False
