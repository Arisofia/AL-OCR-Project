"""
Drift Detection Service for OCR deployment monitoring.
Uses evidently to compare production data against training reference.
"""

import logging
import os
from typing import Optional

import pandas as pd

from ocr_service.config import get_settings

DATA_DRIFT_PRESET_CLS = None  # type: ignore[assignment]
REPORT_CLS = None  # type: ignore[assignment]
TEST_SUITE_CLS = None  # type: ignore[assignment]
TEST_DRIFTED_COLUMNS_CLS = None  # type: ignore[assignment]

try:
    from evidently.metric_preset import DataDriftPreset as _DataDriftPreset
    from evidently.report import Report as _Report
    from evidently.test_suite import TestSuite as _TestSuite
    from evidently.tests import (
        TestNumberOfDriftedColumns as _TestNumberOfDriftedColumns,
    )

    DATA_DRIFT_PRESET_CLS = _DataDriftPreset
    REPORT_CLS = _Report
    TEST_SUITE_CLS = _TestSuite
    TEST_DRIFTED_COLUMNS_CLS = _TestNumberOfDriftedColumns
except ImportError:
    pass

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
