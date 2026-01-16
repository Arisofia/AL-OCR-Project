"""
This file was renamed from logging.py to custom_logging.py to avoid shadowing
the standard library logging module.
"""

import logging  # noqa: E1101  # pylint: disable=E1101
import sys
from pythonjsonlogger import jsonlogger


def setup_logging(level=logging.INFO):
    """
    Configures structured JSON logging for the application.
    """
    handler = logging.StreamHandler(sys.stdout)  # noqa: E1101

    # Define the fields to be included in the JSON output
    fmt = (
        "%(asctime)s %(name)s %(levelname)s %(message)s "
        "%(request_id)s %(method)s %(path)s"
    )
    formatter = jsonlogger.JsonFormatter(fmt=fmt, datefmt="%Y-%m-%dT%H:%M:%S")

    handler.setFormatter(formatter)

    root_logger = logging.getLogger()  # noqa: E1101
    # Remove existing handlers to avoid duplicate logs
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Set some specific loggers to higher levels to reduce noise
    logging.getLogger("boto3").setLevel(logging.WARNING)  # noqa: E1101
    logging.getLogger("botocore").setLevel(logging.WARNING)  # noqa: E1101
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # noqa: E1101
