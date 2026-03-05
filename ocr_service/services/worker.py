"""Worker service for handling S3-triggered Textract processing flows."""

from dataclasses import dataclass
import logging
import os
import urllib.parse
from typing import Any, Optional

from botocore.exceptions import ClientError

from ocr_service.config import Settings, get_settings
from ocr_service.services.storage import StorageService
from ocr_service.services.textract import TextractService, TextractServiceError

logger = logging.getLogger("ocr-service.worker")


@dataclass(frozen=True)
class ProcessingContext:
    """Immutable context for handling one S3 record processing flow."""

    bucket: str
    key: str
    out_key: str
    request_id: str
    storage: StorageService


class WorkerService:
    """
    Handles event-driven background processing (e.g. S3 Triggers).
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.textract_service = TextractService()

    def process_s3_record(
        self, record: dict[str, Any], request_id: str = "N/A"
    ) -> None:
        """
        Processes an individual S3 document record.
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

        storage_service = StorageService(bucket_name=bucket)

        out_key = (
            f"{self.settings.output_prefix.rstrip('/')}/{os.path.basename(key)}.json"
        )
        context = ProcessingContext(
            bucket=bucket,
            key=key,
            out_key=out_key,
            request_id=request_id,
            storage=storage_service,
        )

        try:
            if key.lower().endswith(".pdf"):
                output = self._handle_async_pdf(bucket, key, request_id)
            else:
                output = self._handle_sync_image(bucket, key, request_id)

            if not storage_service.save_json(output, out_key):
                logger.error(
                    "Storage failure: Could not persist extraction for %s", key
                )
            else:
                logger.info("Result persisted | Path: s3://%s/%s", bucket, out_key)

        except ClientError as e:
            self._handle_aws_error(e, context)
            raise
        except TextractServiceError as e:
            self._handle_textract_error(e, context)
            raise
        except Exception as e:
            self._handle_generic_error(e, context)
            raise

    def process_s3_event(
        self, event: dict[str, Any], request_id: str = "N/A"
    ) -> None:
        """Process all records from an S3 event payload."""
        for record in event.get("Records", []):
            self.process_s3_record(record, request_id=request_id)

    def _handle_async_pdf(
        self, bucket: str, key: str, request_id: str
    ) -> dict[str, Any]:
        job_id = self.textract_service.start_detection(bucket, key)
        if not job_id:
            raise RuntimeError(f"Textract job initiation failure for {key}")

        logger.info("Async processing initiated | JobId: %s | Key: %s", job_id, key)
        return {
            "jobId": job_id,
            "status": "STARTED",
            "requestId": request_id,
            "input": {"bucket": bucket, "key": key},
        }

    def _handle_sync_image(
        self, bucket: str, key: str, request_id: str
    ) -> dict[str, Any]:
        output = self.textract_service.analyze_document(bucket, key)
        output["requestId"] = request_id
        logger.info("Sync analysis completed | Key: %s", key)
        return output

    def _handle_aws_error(self, e: ClientError, context: ProcessingContext):
        req_id = e.response.get("ResponseMetadata", {}).get("RequestId", "N/A")
        logger.error(
            "AWS Service Failure | Key: %s | RID: %s | Error: %s",
            context.key,
            req_id,
            e,
        )
        err_obj = {
            "error": "aws_service_failure",
            "message": str(e),
            "requestId": req_id,
            "input": {"bucket": context.bucket, "key": context.key},
        }
        context.storage.save_json(err_obj, context.out_key)

    def _handle_textract_error(
        self,
        e: TextractServiceError,
        context: ProcessingContext,
    ):
        logger.error("Textract processing failed for key %s: %s", context.key, e)
        err_obj = {
            "error": "textract_service_failure",
            "message": str(e),
            "requestId": context.request_id,
            "input": {"bucket": context.bucket, "key": context.key},
        }
        context.storage.save_json(err_obj, context.out_key)

    def _handle_generic_error(
        self,
        e: Exception,
        context: ProcessingContext,
    ):
        logger.exception(
            "Operational failure: Object %s in bucket %s",
            context.key,
            context.bucket,
        )
        err_obj = {
            "error": "internal_pipeline_failure",
            "message": str(e),
            "requestId": context.request_id,
            "input": {"bucket": context.bucket, "key": context.key},
        }
        try:
            context.storage.save_json(err_obj, context.out_key)
        except Exception as se:
            logger.error("Critical storage failure during error logging: %s", se)
