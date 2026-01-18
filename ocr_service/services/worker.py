import logging
import os
import urllib.parse
from typing import Any, Optional

from botocore.exceptions import ClientError

from ocr_service.config import Settings, get_settings
from ocr_service.services.storage import StorageService
from ocr_service.services.textract import TextractService

logger = logging.getLogger("ocr-service.worker")


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

        # Standardize output naming convention
        out_key = (
            f"{self.settings.output_prefix.rstrip('/')}/{os.path.basename(key)}.json"
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
            self._handle_aws_error(e, bucket, key, out_key, storage_service)
            raise
        except Exception as e:
            self._handle_generic_error(
                e, bucket, key, out_key, request_id, storage_service
            )
            raise

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

    def _handle_aws_error(
        self,
        e: ClientError,
        bucket: str,
        key: str,
        out_key: str,
        storage: StorageService,
    ):
        req_id = e.response.get("ResponseMetadata", {}).get("RequestId", "N/A")
        logger.error(
            "AWS Service Failure | Key: %s | RID: %s | Error: %s", key, req_id, e
        )
        err_obj = {
            "error": "aws_service_failure",
            "message": str(e),
            "requestId": req_id,
            "input": {"bucket": bucket, "key": key},
        }
        storage.save_json(err_obj, out_key)

    def _handle_generic_error(
        self,
        e: Exception,
        bucket: str,
        key: str,
        out_key: str,
        request_id: str,
        storage: StorageService,
    ):
        logger.exception("Operational failure: Object %s in bucket %s", key, bucket)
        err_obj = {
            "error": "internal_pipeline_failure",
            "message": str(e),
            "requestId": request_id,
            "input": {"bucket": bucket, "key": key},
        }
        try:
            storage.save_json(err_obj, out_key)
        except Exception as se:
            logger.error("Critical storage failure during error logging: %s", se)
