"""
Drift Detection Service for OCR deployment monitoring.
Uses evidently to compare production data against training reference.
"""

import logging
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns

logger = logging.getLogger("ocr-service.drift")

def check_for_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, report_path: str = "reports/drift_report.html") -> bool:
    """
    Compares 'reference_data' (training set) vs 'current_data' (production inference).
    Focuses on metadata drift and model confidence drift.
    Returns True if significant drift is detected.
    """
    try:
        # 1. Generate Drift Report (for human visualization)
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(reference_data=reference_data, current_data=current_data)
        
        # Ensure directory exists for report
        import os
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        drift_report.save_html(report_path)
        
        # 2. Run Automated Test Suite
        data_test = TestSuite(tests=[TestNumberOfDriftedColumns()])
        data_test.run(reference_data=reference_data, current_data=current_data)
        
        summary = data_test.as_dict()["summary"]
        if not summary["all_passed"]:
            logger.warning("DRIFT DETECTED! Significant divergence in input data distribution.")
            return True
            
        logger.info("No significant drift detected.")
        return False
    except Exception as e:
        logger.exception("Drift detection failed: %s", e)
        return False
