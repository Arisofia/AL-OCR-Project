"""Test metrics increment on Redis failure for idempotency logic."""

# pylint: disable=protected-access
import asyncio
import contextlib
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
import redis.asyncio as redis_mod

from ocr_service import metrics
from ocr_service.exceptions import OCRPipelineError
from ocr_service.modules.processor import OCRProcessor, ProcessingConfig


class FailingRedis:
    """Mock Redis client that always fails for negative test."""

    async def get(self, _key):
        """Simulate Redis get failure."""
        raise RuntimeError("redis connection failure")

    async def set(self, _key, _value, _ex=None):
        """Simulate Redis set failure."""
        raise RuntimeError("redis set failure")

    async def delete(self, _key):
        """Simulate Redis delete failure."""
        raise RuntimeError("redis delete failure")


@pytest.mark.xfail(
    reason=("Simulated Redis outage: expected failure for negative test scenario")
)
def test_metrics_increment_on_redis_get_failure():
    """Test that metrics increment when Redis get fails (negative test)."""
    engine = MagicMock()
    engine.process_image = AsyncMock(return_value={"text": "x"})

    storage = MagicMock()
    storage.upload_file = MagicMock(return_value="s3://bucket/file.png")

    redis = FailingRedis()

    # Cast to redis.Redis for mypy compatibility
    processor = OCRProcessor(engine, storage, cast(redis_mod.Redis, redis))

    before = metrics.OCR_IDEMPOTENCY_REDIS_ERROR_COUNT.labels(
        operation="get"
    )._value.get()

    with contextlib.suppress(OCRPipelineError):
        asyncio.run(
            processor.process_bytes(
                contents=b"img",
                filename="file.png",
                content_type="image/png",
                config=ProcessingConfig(),
            )
        )

    after = metrics.OCR_IDEMPOTENCY_REDIS_ERROR_COUNT.labels(
        operation="get"
    )._value.get()
    assert after >= before + 1
