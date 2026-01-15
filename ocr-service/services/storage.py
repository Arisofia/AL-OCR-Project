"""
Service for interacting with Amazon S3.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

import boto3  # type: ignore
from botocore.config import Config  # type: ignore
from botocore.exceptions import ClientError  # type: ignore
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger("ocr-service.storage")


class StorageService:
    """
    Simple S3 wrapper with basic retry semantics.

    Retries are performed locally on transient ClientError exceptions (e.g. throttling).
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        settings: Optional[Any] = None,
    ):
        """
        Initializes the StorageService with S3 client and configuration.
        """
        if not settings:
            from config import get_settings

            settings = get_settings()

        self.bucket_name = bucket_name or getattr(settings, "s3_bucket_name", None)
        # Use explicit fallback if the setting is None or falsy
        self.max_retries = getattr(settings, "aws_max_retries", None) or 3
        self.region = getattr(settings, "aws_region", "us-east-1")
        self._last_check_time = 0.0
        self._last_check_result = False

        # Disable botocore automatic retries when we use a local manual retry loop
        config = Config(retries={"max_attempts": 1, "mode": "standard"})
        self.s3_client = (
            boto3.client(
                "s3",
                config=config,
                region_name=self.region,
            )
            if self.bucket_name
            else None
        )

    def check_connection(self) -> bool:
        """
        Validates S3 connectivity by checking if the bucket exists.
        Results are cached for 60 seconds to avoid redundant API calls.
        """
        if not self.s3_client or not self.bucket_name:
            return False

        # Return cached result if fresh
        if time.time() - self._last_check_time < 60:
            return self._last_check_result

        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            self._last_check_result = True
        except (ClientError, Exception):
            self._last_check_result = False

        self._last_check_time = time.time()
        return self._last_check_result

    def generate_presigned_post(
        self,
        key: str,
        content_type: str,
        expires_in: int = 3600,
    ) -> Dict[str, Any]:
        """
        Generates a presigned POST URL for S3 upload.
        """
        if not self.s3_client or not self.bucket_name:
            raise RuntimeError("S3 Storage not properly configured for presigning")

        try:
            return self.s3_client.generate_presigned_post(
                Bucket=self.bucket_name,
                Key=key,
                Fields={"Content-Type": content_type},
                Conditions=[["starts-with", "$Content-Type", content_type]],
                ExpiresIn=expires_in,
            )
        except ClientError as e:
            logger.error("Failed to generate presigned POST: %s", e)
            raise

    def upload_file(
        self,
        content: bytes,
        filename: str,
        content_type: str,
        prefix: str = "processed",
    ) -> Optional[str]:
        """
        Upload binary content to S3 and return the object key or None on failure.
        """
        if not self.s3_client or not self.bucket_name:
            logger.warning("S3 Bucket not configured, skipping upload.")
            return None

        s3_key = f"{prefix}/{uuid.uuid4()}-{filename}"
        success = self.put_object(s3_key, content, content_type)
        if success:
            return s3_key
        logger.error("upload_file failed after retries")
        return None

    def upload_json(
        self, data: Any, filename: str, prefix: str = "recon_meta"
    ) -> Optional[str]:
        """
        Serializes data to JSON and uploads it to S3.
        """
        s3_key = f"{prefix}/{uuid.uuid4()}-{filename}.json"
        if self.save_json(data, s3_key):
            return s3_key
        return None

    def save_json(self, data: Any, key: str) -> bool:
        """
        Saves a JSON-serializable object to S3 with retries.
        """
        try:
            body = json.dumps(data).encode("utf-8")
            return self.put_object(key, body, "application/json")
        except (TypeError, ValueError) as e:
            logger.error("Failed to serialize JSON: %s", e)
            return False

    def put_object(self, key: str, body: bytes, content_type: str) -> bool:
        """
        Put an object into S3 with tenacity retry logic for transient errors.
        """
        if not self.s3_client or not self.bucket_name:
            logger.debug("No s3 client or bucket configured; put_object skipped")
            return False

        try:

            @retry(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=0.1, min=0.1, max=2),
                retry=retry_if_exception_type(ClientError),
                reraise=True,
            )
            def _do_put():
                logger.debug(
                    "Attempting to put object to S3: bucket=%s, key=%s",
                    self.bucket_name,
                    key,
                )
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=body,
                    ContentType=content_type,
                )
                logger.info("Successfully put object to S3: key=%s", key)
                return True

            return _do_put()
        except Exception as e:
            logger.error(
                "Exceeded S3 put_object retry attempts or encountered error: %s", e
            )
            return False
