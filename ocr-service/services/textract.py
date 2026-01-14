"""
Amazon Textract integration service for high-throughput financial document intelligence.
"""

import time
import logging
from typing import Dict, Any, Optional, List

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger("ocr-service.textract")


class TextractService:
    """
    Orchestrates AWS Textract interactions with error handling and backoff.
    """

    MAX_POLL_ATTEMPTS = 30
    POLL_INTERVAL_SECONDS = 2

    def __init__(self, settings: Optional[Any] = None):
        if not settings:
            from config import get_settings
            settings = get_settings()

        self.max_retries = getattr(settings, "aws_max_retries", 3)

        # Optimize botocore configuration for deterministic retry management
        config = Config(retries={"max_attempts": 1, "mode": "standard"})
        self.client = boto3.client(
            "textract",
            config=config,
            region_name=getattr(settings, "aws_region", "us-east-1"),
        )

    def start_detection(self, bucket: str, key: str) -> Optional[str]:
        """Initiates async text detection. Returns JobId or None on failure."""
        attempt = 0
        while attempt < self.max_retries:
            try:
                resp = self.client.start_document_text_detection(
                    DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}}
                )
                return resp.get("JobId")
            except ClientError as e:
                attempt += 1
                request_id = e.response.get("ResponseMetadata", {}).get("RequestId")
                logger.warning(
                    "Start detection failed | Attempt: %s | RequestId: %s | Error: %s",
                    attempt,
                    request_id,
                    e,
                )
                time.sleep(0.1 * (2 ** (attempt - 1)))
            except Exception as e:
                logger.exception("Unexpected error in start_detection: %s", e)
                break
        return None

    def analyze_document(
        self, bucket: str, key: str, features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Performs synchronous document analysis for real-time processing pipelines."""
        if features is None:
            features = ["TABLES", "FORMS"]

        attempt = 0
        while attempt < self.max_retries:
            try:
                return self.client.analyze_document(
                    Document={"S3Object": {"Bucket": bucket, "Name": key}},
                    FeatureTypes=features,
                )
            except ClientError as e:
                attempt += 1
                request_id = e.response.get("ResponseMetadata", {}).get("RequestId")
                logger.warning(
                    "Analyze document failed | Attempt: %s | RequestId: %s | Error: %s",
                    attempt,
                    request_id,
                    e,
                )
                time.sleep(0.1 * (2 ** (attempt - 1)))
            except Exception as e:
                logger.exception("Unexpected error in analyze_document: %s", e)
                raise RuntimeError("Critical Textract failure") from e
        raise RuntimeError("Service failure: Max retry threshold reached")

    def get_job_results(self, job_id: str) -> Dict[str, Any]:
        """Polls for asynchronous job completion and aggregates paginated results."""
        attempt = 0
        while attempt < self.MAX_POLL_ATTEMPTS:
            try:
                resp = self.client.get_document_text_detection(JobId=job_id)
                status = resp.get("JobStatus")
                if status == "SUCCEEDED":
                    return self._collect_all_pages(job_id, resp)
                if status == "FAILED":
                    rid = resp.get("ResponseMetadata", {}).get("RequestId")
                    raise RuntimeError(f"Job failed | JobId: {job_id} | RID: {rid}")

                logger.info("Polling | JobId: %s | Status: %s", job_id, status)
                time.sleep(self.POLL_INTERVAL_SECONDS)
                attempt += 1
            except ClientError as e:
                rid = e.response.get("ResponseMetadata", {}).get("RequestId")
                logger.error("Failed | JobId: %s | RID: %s | Error: %s", job_id, rid, e)
                raise
        raise RuntimeError(f"Timeout: Result aggregation exceeded for {job_id}")

    def _collect_all_pages(
        self,
        job_id: str,
        first_response: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregates all paginated blocks using native SDK paginators."""
        blocks = first_response.get("Blocks", [])
        next_token = first_response.get("NextToken")

        if not next_token:
            return first_response

        paginator = self.client.get_paginator("get_document_text_detection")
        page_iterator = paginator.paginate(
            JobId=job_id, PaginationConfig={"StartingToken": next_token}
        )

        for page in page_iterator:
            blocks.extend(page.get("Blocks", []))

        first_response["Blocks"] = blocks
        first_response.pop("NextToken", None)
        return first_response
