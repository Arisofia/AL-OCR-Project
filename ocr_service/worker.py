import json
import logging
import os
import time
import redis  # type: ignore

redis_host = os.getenv("REDIS_HOST", "localhost")
r = redis.Redis(host=redis_host, port=6379, db=0)


def process_queue():
    """
    Main loop for processing OCR jobs from the Redis queue.
    """
    logger = logging.getLogger("ocr-worker")
    logger.info("Worker started, waiting for tasks on 'ocr_tasks'...")

    while True:
        try:
            # Use blocking pop to avoid high CPU usage
            task = r.blpop("ocr_tasks", timeout=0)
            if not task:
                continue

            _, job_id_bytes = task
            job_id = job_id_bytes.decode("utf-8")

            job_key = f"job:{job_id}"
            job_data = r.get(job_key)
            if not job_data:
                logger.warning("Job data not found for ID: %s", job_id)
                continue

            data = json.loads(job_data)
            data["status"] = "PROCESSING"
            r.set(job_key, json.dumps(data))

            logger.info("Processing job: %s", job_id)

            # --- HEAVY OCR INFERENCE HERE ---
            # In a real scenario, we would download the image and use the model
            result = {"text": "Sample Extracted Text", "confidence": 0.95}
            # --------------------------------

            data["status"] = "COMPLETED"
            data["result"] = result
            r.set(job_key, json.dumps(data))
            logger.info("Job %s completed", job_id)

        except Exception as e:
            logger.error("Error processing queue: %s", e)
            # Prevent rapid looping on persistent errors
            time.sleep(1)


if __name__ == "__main__":
    process_queue()
