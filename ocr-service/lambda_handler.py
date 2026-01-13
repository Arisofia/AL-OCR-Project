import boto3
import json
import logging
import os
import urllib.parse

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
textract = boto3.client('textract')

OUTPUT_PREFIX = os.getenv('OUTPUT_PREFIX', 'textract_outputs/')


def handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")

    for record in event.get('Records', []):
        s3_info = record.get('s3', {})
        bucket = s3_info.get('bucket', {}).get('name')
        key = urllib.parse.unquote_plus(s3_info.get('object', {}).get('key', ''))

        if not bucket or not key:
            logger.warning('Missing bucket or key in event record')
            continue

        try:
            if key.lower().endswith('.pdf'):
                # Async job for large PDFs
                resp = textract.start_document_text_detection(
                    DocumentLocation={'S3Object': {'Bucket': bucket, 'Name': key}}
                )
                job_id = resp.get('JobId')
                output = {'jobId': job_id, 'status': 'STARTED', 'input': {'bucket': bucket, 'key': key}}
                write_output(bucket, key, output)
                logger.info(f"Started Textract job {job_id} for {key}")

            else:
                # Synchronous analyze for images
                resp = textract.analyze_document(
                    Document={'S3Object': {'Bucket': bucket, 'Name': key}},
                    FeatureTypes=['TABLES', 'FORMS']
                )
                write_output(bucket, key, resp)
                logger.info(f"Completed Textract analyze for {key}")

        except Exception as e:
            logger.exception(f"Error processing object {key} from bucket {bucket}: {e}")
            err_obj = {'error': str(e), 'input': {'bucket': bucket, 'key': key}}
            write_output(bucket, key, err_obj)

    return {'status': 'ok'}


def write_output(bucket, key, data):
    out_key = os.path.join(OUTPUT_PREFIX, f"{os.path.basename(key)}.json")
    s3.put_object(Body=json.dumps(data), Bucket=bucket, Key=out_key)
    logger.info(f"Wrote output to s3://{bucket}/{out_key}")