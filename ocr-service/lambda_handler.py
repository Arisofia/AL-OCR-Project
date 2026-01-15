"""
Event-driven AWS Lambda entry point for automated document intelligence processing.
Orchestrates S3 lifecycle events to trigger multi-page Textract analysis and
persistence.
"""

import logging
import os
import urllib.parse
from typing import Any, Dict, Tuple

from sentry_sdk.integrations.aws_lambda import AwsLambdaIntegration
from botocore.exceptions import ClientError  # type: ignore

from config import get_settings
from services.storage import StorageService
from services.textract import TextractService
from utils.monitoring import init_monitoring

settings = get_settings()

# Initialize enterprise-grade monitoring (Logging + Sentry)
init_monitoring(settings, integrations=[AwsLambdaIntegration()])
logger = logging.getLogger("ocr-service.lambda")


def get_services(bucket_name: str) -> Tuple[TextractService, StorageService]:
    """
    Dependency factory for AWS integration services.
    """
    return TextractService(), StorageService(bucket_name=bucket_name)


def process_record(record: Dict[str, Any], request_id: str = "N/A") -> None:
    """
    Processes an individual S3 document record with error isolation and traceability.
    """
    s3_info = record.get("s3", {})
    bucket = s3_info.get("bucket", {}).get("name")
    key = urllib.parse.unquote_plus(s3_info.get("object", {}).get("key", ""))

    if not bucket or not key:
        logger.warning(
            "Payload error: Missing S3 bucket or key reference",
            extra={"request_id": request_id},
        )
        return

    textract_service, storage_service = get_services(bucket)

    # Standardize output naming convention for downstream consumption
    out_key = f"{settings.output_prefix.rstrip('/')}/{os.path.basename(key)}.json"

    try:
        # Route documents based on format requirements (Async for PDFs, Sync for images)
        if key.lower().endswith(".pdf"):
            job_id = textract_service.start_detection(bucket, key)
            if not job_id:
                raise RuntimeError(f"Textract job initiation failure for {key}")

            output = {
                "jobId": job_id,
                "status": "STARTED",
                "requestId": request_id,
                "input": {"bucket": bucket, "key": key},
            }
            logger.info(
                "Async processing initiated",
                extra={"job_id": job_id, "key": key, "request_id": request_id},
            )
        else:
            output = textract_service.analyze_document(bucket, key)
            output["requestId"] = request_id
            logger.info(
                "Sync analysis completed", extra={"key": key, "request_id": request_id}
            )

        # Persist extracted intelligence to enterprise storage
        saved = storage_service.save_json(output, out_key)
        if not saved:
            logger.error(
                "Storage failure: Could not persist extraction",
                extra={"key": key, "request_id": request_id},
            )
        else:
            logger.info(
                "Result persisted",
                extra={
                    "path": f"s3://{bucket}/{out_key}",
                    "key": key,
                    "request_id": request_id,
                },
            )

    except ClientError as e:
        aws_request_id = e.response.get("ResponseMetadata", {}).get("RequestId", "N/A")
        logger.error(
            "AWS Service Failure",
            extra={
                "key": key,
                "aws_request_id": aws_request_id,
                "request_id": request_id,
                "error": str(e),
            },
        )
        err_obj: Dict[str, Any] = {
            "error": "aws_service_failure",
            "message": str(e),
            "requestId": aws_request_id,
            "input": {"bucket": bucket, "key": key},
        }
        storage_service.save_json(err_obj, out_key)
        raise
    except Exception as e:
        logger.exception(
            "Operational failure",
            extra={"key": key, "bucket": bucket, "request_id": request_id},
        )
        err_obj = {
            "error": "internal_pipeline_failure",
            "message": str(e),
            "requestId": request_id,  # From process_record argument
            "input": {"bucket": bucket, "key": key},
        }
        try:
            storage_service.save_json(err_obj, out_key)
        except Exception as se:
            logger.error(
                "Critical storage failure during error logging",
                extra={"error": str(se), "request_id": request_id},
            )

        # Propagate exception for Lambda retry visibility
        raise


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for orchestrating S3 document triggers.
    Ensures full traceability and partial failure reporting.
    """
    request_id = getattr(context, "aws_request_id", "local-test")
    records = event.get("Records", [])
    logger.info(
        "Lambda trigger",
        extra={"request_id": request_id, "record_count": len(records)},
    )

    failures = 0
    for record in records:
        try:
            process_record(record, request_id=request_id)
        except Exception:
            failures += 1

    if failures:
        logger.warning(
            "Batch completion with partial failures",
            extra={"request_id": request_id, "failed_count": failures},
        )
        return {"status": "partial_failure", "failed": failures}

    return {"status": "ok"}
