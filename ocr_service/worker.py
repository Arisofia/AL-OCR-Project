"""Asynchronous Redis-backed OCR worker for queued document processing."""

import asyncio
import json
import logging
import time
from typing import Any, Optional

import redis.asyncio as redis

from ocr_service.config import Settings, get_settings
from ocr_service.models import JobStatus
from ocr_service.modules.ocr_config import EngineConfig
from ocr_service.modules.ocr_engine import IterativeOCREngine
from ocr_service.utils.image import decode_image, load_image_from_path
from ocr_service.utils.monitoring import init_monitoring
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
            ocr_strategy_profile=self.settings.ocr_strategy_profile,
            enable_bin_lookup=self.settings.enable_bin_lookup,
        )
        self.engine = IterativeOCREngine(config=engine_config)
        self.queue_name = "ocr_tasks"

    async def start(self):
        """Starts the main worker loop."""
        logger.info("Redis worker started | Queue: %s", self.queue_name)

        while True:
            try:
                # Blocking pop from the queue
                task = await self.redis_client.blpop(  # type: ignore[misc]
                    [self.queue_name], timeout=5
                )
                if not task:
                    continue

                _, job_id_bytes = task
                job_id = job_id_bytes.decode("utf-8")

                await self.process_job(job_id)

            except redis.ConnectionError:
                logger.error("Redis connection lost | Retrying in 5s...")
                await asyncio.sleep(5)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.exception("Unexpected error in worker loop: %s", e)
                await asyncio.sleep(1)

    async def process_job(self, job_id: str):
        """Processes a single OCR job from Redis."""
        job_key = f"job:{job_id}"
        idempotency_key = f"idempotency:{job_id}"
        request_id = "N/A"

        try:
            # Atomic claim and status check
            initial_status = {
                "status": JobStatus.PROCESSING,
                "updated_at": time.time(),
                "request_id": request_id,
            }

            # Try to claim the job using SET NX
            claimed = await self.redis_client.set(
                idempotency_key, json.dumps(initial_status), nx=True, ex=3600
            )

            if not claimed:
                # Job already exists in idempotency cache, check status
                cached_status_raw = await self.redis_client.get(idempotency_key)
                if cached_status_raw:
                    cached_data = json.loads(cached_status_raw)
                    if cached_data.get("status") in [
                        JobStatus.COMPLETED,
                        JobStatus.PROCESSING,
                    ]:
                        logger.info(
                            "Job %s is already %s, skipping",
                            job_id,
                            cached_data.get("status"),
                        )
                        return

                # If FAILED or missing, try to reclaim (NX prevented above)
                # For simplicity, we just return if we couldn't claim it.
                return

            job_data_raw = await self.redis_client.get(job_key)
            if not job_data_raw:
                logger.warning("Job data not found for job_id: %s", job_id)
                await self.redis_client.delete(idempotency_key)
                return

            job_data = json.loads(job_data_raw.decode("utf-8"))
            request_id = job_data.get("request_id", request_id)

            # Update idempotency with correct request_id
            initial_status["request_id"] = request_id
            await self.redis_client.set(
                idempotency_key, json.dumps(initial_status), ex=3600
            )

            logger.info("Processing job | ID: %s | RID: %s", job_id, request_id)

            # OCR Execution
            result = await self._execute_ocr(job_data)

            # Completion
            final_status = {
                "status": JobStatus.COMPLETED,
                "result": result,
                "completed_at": time.time(),
                "request_id": request_id,
            }
            await self.redis_client.set(job_key, json.dumps(final_status))
            await self.redis_client.set(
                idempotency_key,
                json.dumps(final_status),
                ex=self.settings.redis_idempotency_ttl,
            )

            logger.info("Job completed | ID: %s | RID: %s", job_id, request_id)

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.exception("Job failed | ID: %s | RID: %s", job_id, request_id)
            await self._handle_job_failure(job_key, job_id, e, request_id)

            error_status = {
                "status": JobStatus.FAILED,
                "error": str(e),
                "request_id": request_id,
                "failed_at": time.time(),
            }
            await self.redis_client.set(
                idempotency_key, json.dumps(error_status), ex=300
            )

    async def _execute_ocr(self, job_data: dict[str, Any]) -> dict[str, Any]:
        """Runs the actual OCR engine on the provided data."""
        image_bytes: Optional[bytes] = None

        if "image_bytes" in job_data:
            image_bytes = decode_image(job_data["image_bytes"])
        elif "image_path" in job_data:
            image_bytes = load_image_from_path(job_data["image_path"])

        if not image_bytes:
            return {
                "text": "No valid image data provided",
                "confidence": 0.0,
                "error": "missing_input",
            }

        return await self.engine.process_image(
            image_bytes,
            self.settings.enable_reconstruction,
        )

    async def _handle_job_failure(
        self, job_key: str, job_id: str, error: Exception, request_id: str
    ):
        """Handles job failure by updating status in Redis."""
        try:
            job_data_raw = await self.redis_client.get(job_key)
            if job_data_raw:
                job_data = json.loads(job_data_raw.decode("utf-8"))
                job_data.update(
                    {
                        "status": JobStatus.FAILED,
                        "error": str(error),
                        "failed_at": time.time(),
                        "request_id": request_id,
                    }
                )
                await self.redis_client.set(job_key, json.dumps(job_data))
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to record job failure for ID: %s", job_id)


async def main():
    """Initialize monitoring and run the Redis OCR worker loop."""
    settings = get_settings()
    init_monitoring(settings)
    worker = RedisWorker(settings=settings)
    await worker.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
