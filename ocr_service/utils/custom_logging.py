"""
This file was renamed from logging.py to custom_logging.py to avoid shadowing
the standard library logging module.
"""

import logging
import sys

from pythonjsonlogger.json import JsonFormatter


def setup_logging(level=logging.INFO):
    """
    Configures structured JSON logging for the application.
    """
    handler = logging.StreamHandler(sys.stdout)

    # Define the fields to be included in the JSON output
    fmt = (
        "%(asctime)s %(name)s %(levelname)s %(message)s "
        "%(request_id)s %(method)s %(path)s"
    )
    formatter = JsonFormatter(fmt=fmt, datefmt="%Y-%m-%dT%H:%M:%S")

    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Remove existing handlers to avoid duplicate logs
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Set some specific loggers to higher levels to reduce noise
    for name in ("boto3", "botocore", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.WARNING)
