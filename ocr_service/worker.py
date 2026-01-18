import asyncio
import json
import logging
import time
from typing import Any, Optional

import redis.asyncio as redis

from ocr_service.config import Settings, get_settings
from ocr_service.modules.ocr_config import EngineConfig
from ocr_service.modules.ocr_engine import IterativeOCREngine
from ocr_service.utils.custom_logging import setup_logging
from ocr_service.utils.redis_factory import get_redis_client

logger = logging.getLogger("ocr-service.redis-worker")


class RedisWorker:
    """
    Production-grade Redis queue worker for asynchronous OCR tasks.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        redis_client: Optional[redis.Redis] = None,
    ):
        self.settings = settings or get_settings()
        self.redis_client = redis_client or get_redis_client(self.settings)

        # Initialize OCR engine with configuration from settings
        engine_config = EngineConfig(
            max_iterations=self.settings.ocr_iterations,
            enable_reconstruction=self.settings.enable_reconstruction,
        )
        self.engine = IterativeOCREngine(config=engine_config)
        self.queue_name = "ocr_tasks"

    async def start(self):
        """Starts the main worker loop."""
        logger.info("Redis worker started | Queue: %s", self.queue_name)

        while True:
            try:
                # Blocking pop from the queue
                task = await self.redis_client.blpop([self.queue_name], timeout=5)  # type: ignore[misc]
                if not task:
                    continue

                _, job_id_bytes = task
                job_id = job_id_bytes.decode("utf-8")

                await self.process_job(job_id)

            except redis.ConnectionError:
                logger.error("Redis connection lost | Retrying in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.exception("Unexpected error in worker loop: %s", e)
                await asyncio.sleep(1)

    async def process_job(self, job_id: str):
        """Processes a single OCR job from Redis."""
        job_key = f"job:{job_id}"
        job_data_raw = await self.redis_client.get(job_key)

        if not job_data_raw:
            logger.warning("Job data not found | ID: %s", job_id)
            return

        try:
            job_data = json.loads(job_data_raw.decode("utf-8"))

            # Update status to PROCESSING
            job_data["status"] = "PROCESSING"
            job_data["updated_at"] = time.time()
            await self.redis_client.set(job_key, json.dumps(job_data))

            logger.info("Processing job | ID: %s", job_id)

            # --- OCR Extraction Logic ---
            # In a Redis queue context, we expect the image path or raw bytes
            # to be in job_data.
            # For this implementation, we'll look for 'image_path' or 'image_base64'

            result = await self._execute_ocr(job_data)

            # Update status to COMPLETED
            job_data["status"] = "COMPLETED"
            job_data["result"] = result
            job_data["completed_at"] = time.time()
            await self.redis_client.set(job_key, json.dumps(job_data))

            logger.info("Job completed | ID: %s", job_id)

        except Exception as e:
            logger.exception("Job failed | ID: %s | Error: %s", job_id, e)
            await self._handle_job_failure(job_key, job_id, e)

    async def _execute_ocr(self, job_data: dict[str, Any]) -> dict[str, Any]:
        """Runs the actual OCR engine on the provided data."""
        image_bytes: Optional[bytes] = None

        if "image_bytes" in job_data:
            import base64

            try:
                # Handle potential base64 string
                raw_data = job_data["image_bytes"]
                if isinstance(raw_data, str):
                    image_bytes = base64.b64decode(raw_data)
                else:
                    image_bytes = raw_data
            except Exception as e:
                logger.error("Failed to decode image_bytes: %s", e)
                return {"error": "invalid_image_encoding"}

        elif "image_path" in job_data:
            try:
                with open(job_data["image_path"], "rb") as f:
                    image_bytes = f.read()
            except Exception as e:
                logger.error(
                    "Failed to read image_path %s: %s", job_data["image_path"], e
                )
                return {"error": "file_not_found"}

        if not image_bytes:
            logger.warning("No valid image data found in job %s", job_data.get("id"))
            return {
                "text": "No image data provided",
                "confidence": 0.0,
                "error": "missing_input",
            }

        # Use the IterativeOCREngine for high-fidelity processing
        return await self.engine.process_image(
            image_bytes, use_reconstruction=self.settings.enable_reconstruction
        )

    async def _handle_job_failure(self, job_key: str, job_id: str, error: Exception):
        """Handles job failure by updating status in Redis."""
        try:
            job_data_raw = await self.redis_client.get(job_key)
            if job_data_raw:
                job_data = json.loads(job_data_raw.decode("utf-8"))
                job_data["status"] = "FAILED"
                job_data["error"] = str(error)
                job_data["failed_at"] = time.time()
                await self.redis_client.set(job_key, json.dumps(job_data))
        except Exception as e:
            logger.error(
                "Failed to record job failure in Redis | ID: %s | Error: %s", job_id, e
            )


async def main():
    setup_logging(level=logging.INFO)
    worker = RedisWorker()
    await worker.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
