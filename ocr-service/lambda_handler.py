"""
AWS Lambda handler for the OCR service.

This module processes S3 events, triggers Amazon Textract for OCR, 
and saves the results back to S3.
"""

import logging
import os
import urllib.parse
from typing import Dict, Any, Tuple

from services.textract import TextractService
from services.storage import StorageService
from config import get_settings

logger = logging.getLogger("ocr-service.lambda")
logger.setLevel(logging.INFO)

settings = get_settings()


def get_services(bucket_name: str) -> Tuple[TextractService, StorageService]:
    """
    Initializes and returns the Textract and Storage services.
    """
    return TextractService(), StorageService(bucket_name=bucket_name)


def process_record(record: Dict[str, Any]) -> None:
    """
    Processes a single S3 record from the Lambda event.
    """
    s3_info = record.get('s3', {})
    bucket = s3_info.get('bucket', {}).get('name')
    key = urllib.parse.unquote_plus(
        s3_info.get('object', {}).get('key', '')
    )

    if not bucket or not key:
        logger.warning('Missing bucket or key in event record')
        return

    textract_service, storage_service = get_services(bucket)

    # Build a stable S3 key
    out_key = (
        f"{settings.output_prefix.rstrip('/')}/"
        f"{os.path.basename(key)}.json"
    )

    try:
        if key.lower().endswith('.pdf'):
            job_id = textract_service.start_detection(bucket, key)
            if not job_id:
                raise RuntimeError(f"Failed to start Textract detection job for {key}")
            output = {
                'jobId': job_id,
                'status': 'STARTED',
                'input': {'bucket': bucket, 'key': key},
            }
            logger.info("Started Textract job %s for %s", job_id, key)
        else:
            output = textract_service.analyze_document(bucket, key)
            logger.info("Completed Textract analyze for %s", key)

        saved = storage_service.save_json(output, out_key)
        if not saved:
            logger.error(
                "Failed to save Textract output for %s to %s", key, out_key
            )
        else:
            logger.info("Wrote output to s3://%s/%s", bucket, out_key)

    except Exception as e:
        logger.exception(
            "Error processing object %s from bucket %s", key, bucket
        )
        err_obj = {
            'error': f'processing_failed: {str(e)}', 
            'input': {'bucket': bucket, 'key': key}
        }
        storage_service.save_json(err_obj, out_key)


def handler(event: Dict[str, Any], _context: Any) -> Dict[str, str]:
    """
    Main Lambda handler entry point.

    Processes each record in the incoming S3 event.
    """
    records = event.get('Records', [])
    logger.info("Received event with %d record(s)", len(records))
    failures = 0
    for record in records:
        try:
            process_record(record)
        except Exception:
            failures += 1

    if failures:
        logger.warning("Processing completed with %d failure(s)", failures)
        return {"status": "partial_failure", "failed": failures}
    return {"status": "ok"}
