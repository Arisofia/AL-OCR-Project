import json
import logging
import os
import urllib.parse
from typing import Dict, Any

from services.textract import TextractService
from services.storage import StorageService
from config import get_settings

logger = logging.getLogger("ocr-service.lambda")
logger.setLevel(logging.INFO)

settings = get_settings()

def get_services(bucket_name: str):
    return TextractService(), StorageService(bucket_name=bucket_name)

def process_record(record: Dict[str, Any]):
    s3_info = record.get('s3', {})
    bucket = s3_info.get('bucket', {}).get('name')
    key = urllib.parse.unquote_plus(s3_info.get('object', {}).get('key', ''))

    if not bucket or not key:
        logger.warning('Missing bucket or key in event record')
        return

    textract_service, storage_service = get_services(bucket)
    # Build a stable S3 key (avoid os.path.join which can use platform separators)
    out_key = f"{settings.output_prefix.rstrip('/')}/{os.path.basename(key)}.json"

    try:
        if key.lower().endswith('.pdf'):
            job_id = textract_service.start_detection(bucket, key)
            output = {
                'jobId': job_id, 
                'status': 'STARTED', 
                'input': {'bucket': bucket, 'key': key}
            }
            logger.info(f"Started Textract job {job_id} for {key}")
        else:
            output = textract_service.analyze_document(bucket, key)
            logger.info(f"Completed Textract analyze for {key}")

        saved = storage_service.save_json(output, out_key)
        if not saved:
            logger.error(f"Failed to save Textract output for {key} to {out_key}")
        else:
            logger.info(f"Wrote output to s3://{bucket}/{out_key}")

    except Exception as e:
        logger.exception(f"Error processing object {key} from bucket {bucket}: {e}")
        err_obj = {'error': str(e), 'input': {'bucket': bucket, 'key': key}}
        saved = storage_service.save_json(err_obj, out_key)
        if not saved:
            logger.error(f"Failed to save error object for {key} to {out_key}")

def handler(event: Dict[str, Any], context: Any) -> Dict[str, str]:
    logger.info(f"Received event: {json.dumps(event)}")
    for record in event.get('Records', []):
        process_record(record)
    return {'status': 'ok'}
