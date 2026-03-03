"""
Data Quality Gate for OCR lifecycle.
Uses Great Expectations to validate data integrity before training or labeling.
"""

import logging
from importlib import import_module

import pandas as pd

try:
    GX_MODULE = import_module("great_expectations")
except ImportError:
    GX_MODULE = None  # type: ignore[assignment]

logger = logging.getLogger("ocr-service.validation")


def validate_ocr_batch(df: pd.DataFrame) -> bool:
    """
    Validates a batch of OCR data before it enters the AL loop.
    Schema: [image_path, ocr_text, confidence, user_label]
    """
    try:
        if GX_MODULE is None:
            logger.warning("great_expectations is not installed; validation skipped")
            return False

        context = GX_MODULE.get_context()

        # Define Validator
        validator = context.sources.pandas_default.read_dataframe(df)  # type: ignore

        # --- Expectation 1: Critical Columns Exist ---
        # Note: great_expectations API for expect_table_columns_to_match_set or list
        validator.expect_column_to_exist("image_path")
        validator.expect_column_to_exist("ocr_text")
        validator.expect_column_to_exist("confidence")
        validator.expect_column_to_exist("user_label")

        # --- Expectation 2: Confidence Scores must be valid probabilities ---
        validator.expect_column_values_to_be_between(
            "confidence", min_value=0.0, max_value=1.0
        )

        # --- Expectation 3: Image paths must have valid extensions ---
        validator.expect_column_values_to_match_regex(
            "image_path", regex=r".*\.(jpg|jpeg|png|tiff)$"
        )

        # Run Validation
        results = validator.validate()

        if not results["success"]:
            logger.error("Data Validation FAILED!")
            # Log some failure details safely
            return False

        logger.info("Data Validation PASSED.")
        return True
    except (AttributeError, TypeError, ValueError, KeyError) as e:
        logger.exception("Critical error during data validation: %s", e)
        return False
