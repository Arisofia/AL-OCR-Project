"""
Event-driven AWS Lambda entry point for automated document intelligence processing.
"""

import logging
from typing import Any

from ocr_service.services.worker import WorkerService

logger = logging.getLogger("ocr-service.lambda")
logger.setLevel(logging.INFO)

worker = WorkerService()


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Main Lambda handler for orchestrating S3 document triggers.
    """
    request_id = getattr(context, "aws_request_id", "local-test")
    records = event.get("Records", [])
    logger.info("Lambda trigger | RID: %s | Records: %d", request_id, len(records))

    failures = 0
    for record in records:
        try:
            worker.process_s3_record(record, request_id=request_id)
        except Exception:
            failures += 1

    if failures:
        logger.warning("Batch completion with partial failures | Failed: %d", failures)
        return {"status": "partial_failure", "failed": failures}

    return {"status": "ok"}
