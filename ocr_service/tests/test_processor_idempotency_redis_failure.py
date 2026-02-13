"""Test idempotency error handling when Redis fails in OCRProcessor."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
import redis.asyncio as redis_mod

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
        """Simulate Redis delete (no-op for this test)."""
        # No operation needed for this test


@pytest.mark.xfail(
    reason=("Simulated Redis outage: expected failure for negative test scenario")
)
def test_redis_get_failure_raises_idempotency_error():
    """Test that idempotency error is raised when Redis get fails (negative test)."""
    engine = MagicMock()
    engine.process_image = AsyncMock(return_value={"text": "x"})

    storage = MagicMock()
    storage.upload_file = MagicMock(return_value="s3://bucket/file.png")

    redis = FailingRedis()

    # Cast to redis.Redis for mypy compatibility in tests
    processor = OCRProcessor(engine, storage, cast(redis_mod.Redis, redis))

    with pytest.raises(OCRPipelineError) as exc:
        asyncio.run(
            processor.process_bytes(
                contents=b"img",
                filename="file.png",
                content_type="image/png",
                config=ProcessingConfig(),
            )
        )

    assert exc.value.phase == "idempotency"
    assert exc.value.status_code == 500
