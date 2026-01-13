"""
Service for interacting with Amazon Textract.
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
    Wrapper around boto3 textract client with simple error handling and retries.
    """

    # Polling and retry constants
    MAX_POLL_ATTEMPTS = 30
    POLL_INTERVAL_SECONDS = 2

    def __init__(self, settings: Optional[Any] = None):
        if not settings:
            from config import get_settings
            settings = get_settings()

        # Ensure a sane fallback for retries
        self.max_retries = (getattr(settings, "aws_max_retries", None) or 3)

        # When using manual retry loops, keep botocore retries minimal to avoid doubled retries
        config = Config(
            retries={
                'max_attempts': 1,
                'mode': 'standard'
            }
        )
        self.client = boto3.client('textract', config=config, region_name=getattr(settings, "aws_region", "us-east-1"))

    def start_detection(self, bucket: str, key: str) -> Optional[str]:
        """
        Starts asynchronous document text detection and returns JobId or None on failure.
        """
        attempt = 0
        while attempt < self.max_retries:
            try:
                resp = self.client.start_document_text_detection(
                    DocumentLocation={
                        'S3Object': {'Bucket': bucket, 'Name': key}
                    }
                )
                return resp.get('JobId')
            except ClientError as e:
                attempt += 1
                logger.warning("start_detection attempt %s failed: %s", attempt, e)
                time.sleep(0.1 * (2 ** (attempt - 1)))
            except Exception as e:
                logger.exception("Unexpected error during start_detection: %s", e)
                break
        logger.error("start_detection failed after retries")
        return None

    def analyze_document(
        self,
        bucket: str,
        key: str,
        features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Performs synchronous document analysis. Raises RuntimeError on persistent failures.
        """
        if features is None:
            features = ['TABLES', 'FORMS']

        attempt = 0
        while attempt < self.max_retries:
            try:
                return self.client.analyze_document(
                    Document={'S3Object': {'Bucket': bucket, 'Name': key}},
                    FeatureTypes=features
                )
            except ClientError as e:
                attempt += 1
                logger.warning("analyze_document attempt %s failed: %s", attempt, e)
                time.sleep(0.1 * (2 ** (attempt - 1)))
            except Exception as e:
                logger.exception("Unexpected error during analyze_document: %s", e)
                raise RuntimeError("Textract analysis failed") from e
        logger.error("analyze_document failed after retries")
        raise RuntimeError("Textract analysis failed after retries")

    def get_job_results(self, job_id: str) -> Dict[str, Any]:
        """
        Retrieves results of an asynchronous job with polling/waiting.
        """
        attempt = 0
        while attempt < self.MAX_POLL_ATTEMPTS:
            try:
                resp = self.client.get_document_text_detection(JobId=job_id)
                status = resp.get('JobStatus')
                if status == 'SUCCEEDED':
                    return self._collect_all_pages(job_id, resp)
                if status == 'FAILED':
                    raise RuntimeError(f"Textract job {job_id} failed")

                logger.info("Job %s status: %s. Waiting...", job_id, status)
                time.sleep(self.POLL_INTERVAL_SECONDS)
                attempt += 1
            except ClientError as e:
                logger.error("Error getting results for job %s: %s", job_id, e)
                raise
        raise RuntimeError(f"Timeout waiting for job {job_id}")

    def _collect_all_pages(self, job_id: str, first_response: Dict[str, Any]) -> Dict[str, Any]:
        """Collects all pages of results using NextToken pagination."""
        blocks = first_response.get('Blocks', [])
        next_token = first_response.get('NextToken')
        
        while next_token:
            resp = self.client.get_document_text_detection(JobId=job_id, NextToken=next_token)
            blocks.extend(resp.get('Blocks', []))
            next_token = resp.get('NextToken')
        
        first_response['Blocks'] = blocks
        return first_response
