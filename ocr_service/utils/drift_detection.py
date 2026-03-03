"""
Drift Detection Service for OCR deployment monitoring.
Uses evidently to compare production data against training reference.
"""

import logging
import os
from importlib import import_module
from typing import Optional

import pandas as pd

from ocr_service.config import get_settings

DATA_DRIFT_PRESET_CLS = None
REPORT_CLS = None
TEST_SUITE_CLS = None
TEST_DRIFTED_COLUMNS_CLS = None

try:
    _metric_preset_mod = import_module("evidently.metric_preset")
    _report_mod = import_module("evidently.report")
    _test_suite_mod = import_module("evidently.test_suite")
    _tests_mod = import_module("evidently.tests")
except ImportError:
    _metric_preset_mod = None
    _report_mod = None
    _test_suite_mod = None
    _tests_mod = None

if _metric_preset_mod is not None:
    DATA_DRIFT_PRESET_CLS = getattr(_metric_preset_mod, "DataDriftPreset", None)
if _report_mod is not None:
    REPORT_CLS = getattr(_report_mod, "Report", None)
if _test_suite_mod is not None:
    TEST_SUITE_CLS = getattr(_test_suite_mod, "TestSuite", None)
if _tests_mod is not None:
    TEST_DRIFTED_COLUMNS_CLS = getattr(_tests_mod, "TestNumberOfDriftedColumns", None)

logger = logging.getLogger("ocr-service.drift")


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
        # 1. Generate Drift Report (for human visualization)
        drift_report = REPORT_CLS(metrics=[DATA_DRIFT_PRESET_CLS()])
        drift_report.run(reference_data=reference_data, current_data=current_data)

        # Ensure directory exists for report
        os.makedirs(os.path.dirname(actual_report_path), exist_ok=True)
        drift_report.save_html(actual_report_path)

        # 2. Run Automated Test Suite
        data_test = TEST_SUITE_CLS(tests=[TEST_DRIFTED_COLUMNS_CLS()])
        data_test.run(reference_data=reference_data, current_data=current_data)

        summary = data_test.as_dict()["summary"]
        if not summary["all_passed"]:
            logger.warning(
                "DRIFT DETECTED! Significant divergence in input data distribution."
            )
            return True

        logger.info("No significant drift detected.")
        return False
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as e:
        logger.exception("Drift detection failed: %s", e)
        return False
