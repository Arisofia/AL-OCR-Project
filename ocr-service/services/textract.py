import time
import boto3
import logging
from typing import Dict, Any, Optional
from botocore.config import Config
from botocore.exceptions import ClientError
from config import get_settings

logger = logging.getLogger("ocr-service.textract")

class TextractService:
    """Wrapper around boto3 textract client with simple error handling and retries."""

    def __init__(self):
        settings = get_settings()
        self.max_retries = settings.aws_max_retries or 3
        config = Config(
            retries={
                'max_attempts': settings.aws_max_retries,
                'mode': 'standard'
            },
            region_name=settings.aws_region
        )
        self.client = boto3.client('textract', config=config)

    def start_detection(self, bucket: str, key: str) -> Optional[str]:
        """Starts asynchronous document text detection and returns JobId or None on failure."""
        attempt = 0
        while attempt < self.max_retries:
            try:
                resp = self.client.start_document_text_detection(
                    DocumentLocation={'S3Object': {'Bucket': bucket, 'Name': key}}
                )
                return resp.get('JobId')
            except ClientError as e:
                attempt += 1
                logger.warning(f"start_detection attempt {attempt} failed: {e}")
                time.sleep(0.1 * (2 ** (attempt - 1)))
            except Exception as e:
                logger.exception(f"Unexpected error during start_detection: {e}")
                break
        logger.error("start_detection failed after retries")
        return None

    def analyze_document(self, bucket: str, key: str, features: list = ['TABLES', 'FORMS']) -> Dict[str, Any]:
        """Performs synchronous document analysis. Raises RuntimeError on persistent failures."""
        attempt = 0
        while attempt < self.max_retries:
            try:
                return self.client.analyze_document(
                    Document={'S3Object': {'Bucket': bucket, 'Name': key}},
                    FeatureTypes=features
                )
            except ClientError as e:
                attempt += 1
                logger.warning(f"analyze_document attempt {attempt} failed: {e}")
                time.sleep(0.1 * (2 ** (attempt - 1)))
            except Exception as e:
                logger.exception(f"Unexpected error during analyze_document: {e}")
                raise RuntimeError("Textract analysis failed")
        logger.error("analyze_document failed after retries")
        raise RuntimeError("Textract analysis failed after retries")
