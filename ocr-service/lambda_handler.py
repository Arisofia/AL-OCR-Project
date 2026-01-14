"""
Event-driven AWS Lambda entry point for automated document intelligence processing.
Orchestrates S3 lifecycle events to trigger multi-page Textract analysis and
persistence.
"""

import logging
import os
import urllib.parse
from typing import Dict, Any, Tuple

from botocore.exceptions import ClientError
from services.textract import TextractService
from services.storage import StorageService
from config import get_settings

logger = logging.getLogger("ocr-service.lambda")
logger.setLevel(logging.INFO)

settings = get_settings()


def get_services(bucket_name: str) -> Tuple[TextractService, StorageService]:
    """
    Dependency factory for AWS integration services.
    """
    return TextractService(), StorageService(bucket_name=bucket_name)


def process_record(record: Dict[str, Any]) -> None:
    """
    Processes an individual S3 document record with error isolation and traceability.
    """
    s3_info = record.get("s3", {})
    bucket = s3_info.get("bucket", {}).get("name")
    key = urllib.parse.unquote_plus(s3_info.get("object", {}).get("key", ""))

    if not bucket or not key:
        logger.warning("Payload error: Missing S3 bucket or key reference")
        return

    textract_service, storage_service = get_services(bucket)

    # Standardize output naming convention for deterministic downstream consumption
    out_key = (
        f"{settings.output_prefix.rstrip('/')}/"
        f"{os.path.basename(key)}.json"
    )

    try:
        # Route documents based on format requirements (Async for PDFs, Sync for images)
        if key.lower().endswith(".pdf"):
            job_id = textract_service.start_detection(bucket, key)
            if not job_id:
                raise RuntimeError(f"Textract job initiation failure for {key}")

            output = {
                "jobId": job_id,
                "status": "STARTED",
                "input": {"bucket": bucket, "key": key},
                "requestId": "ASYNC_PENDING",
            }
            logger.info("Async processing initiated | JobId: %s | Key: %s", job_id, key)
        else:
            output = textract_service.analyze_document(bucket, key)
            request_id = output.get("ResponseMetadata", {}).get("RequestId", "N/A")
            output["requestId"] = request_id
            logger.info("Sync analysis completed | Key: %s | RequestId: %s", key, request_id)

        # Persist extracted intelligence to enterprise storage
        saved = storage_service.save_json(output, out_key)
        if not saved:
            logger.error("Storage failure: Could not persist extraction for %s", key)
        else:
            logger.info("Result persisted | Path: s3://%s/%s", bucket, out_key)

    except (ClientError, RuntimeError) as e:
        logger.exception("Operational failure: Object %s in bucket %s", key, bucket)
        # Extract AWS request id when available for diagnostics
        request_id = "N/A"
        if isinstance(e, ClientError):
            request_id = e.response.get("ResponseMetadata", {}).get("RequestId", "N/A")

        err_obj = {
            "error": f"extraction_pipeline_failed: {e}",
            "requestId": request_id,
            "input": {"bucket": bucket, "key": key},
        }
        try:
            storage_service.save_json(err_obj, out_key)
        except (ClientError, TypeError, ValueError) as se:
            logger.error("Critical storage failure during error logging: %s", se)

        # Propagate exception for Lambda retry visibility
        raise
    except Exception as e:
        logger.exception("Unexpected system failure: %s", e)
        raise


def handler(event: Dict[str, Any], _context: Any) -> Dict[str, str]:
    """
    Main Lambda handler for orchestrating S3 document triggers.
    Ensures full traceability and partial failure reporting.
    """
    records = event.get("Records", [])
    logger.info("Lambda trigger | Records detected: %d", len(records))

    failures = 0
    for record in records:
        try:
            process_record(record)
        except (ClientError, RuntimeError, ValueError):
            failures += 1
        except Exception:
            logger.exception("Uncaught exception in record processing")
            failures += 1

    if failures:
        logger.warning("Batch completion with partial failures | Failed: %d", failures)
        return {"status": "partial_failure", "failed": failures}

    return {"status": "ok"}
