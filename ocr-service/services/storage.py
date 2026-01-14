"""
Service for interacting with Amazon S3.
"""

import json
import uuid
import logging
import time
from typing import Optional, Any

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

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
        self.max_retries = (getattr(settings, "aws_max_retries", None) or 3)
        self.region = getattr(settings, "aws_region", "us-east-1")

        # Disable botocore automatic retries when we use a local manual retry loop
        config = Config(
            retries={
                'max_attempts': 1,
                'mode': 'standard'
            }
        )
        self.s3_client = (
            boto3.client(
                's3',
                config=config,
                region_name=self.region,
            ) if self.bucket_name else None
        )

    def upload_file(
        self,
        content: bytes,
        filename: str,
        content_type: str,
        prefix: str = "processed"
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
        self,
        data: Any,
        filename: str,
        prefix: str = "recon_meta"
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
        Put an object into S3 with simple retry/backoff for transient errors.
        """
        if not self.s3_client or not self.bucket_name:
            logger.debug("No s3 client or bucket configured; put_object skipped")
            return False

        attempt = 0
        while attempt < self.max_retries:
            try:
                logger.debug(
                    "Attempting to put object to S3: bucket=%s, key=%s, attempt=%s",
                    self.bucket_name,
                    key,
                    attempt + 1
                )
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=body,
                    ContentType=content_type
                )
                logger.info("Successfully put object to S3: key=%s", key)
                return True
            except ClientError as e:
                attempt += 1
                logger.warning("S3 put_object attempt %s failed: %s", attempt, e)
                # exponential backoff (small; safe for unit tests)
                backoff = 0.1 * (2 ** (attempt - 1))
                time.sleep(backoff)
        logger.error("Exceeded S3 put_object retry attempts")
        return False
