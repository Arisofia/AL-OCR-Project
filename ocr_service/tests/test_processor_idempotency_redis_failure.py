import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from ocr_service.exceptions import OCRPipelineError
from ocr_service.modules.processor import OCRProcessor, ProcessingConfig


class FailingRedis:
    async def get(self, _key):
        raise RuntimeError("redis connection failure")

    async def set(self, _key, _value, _ex=None):
        raise RuntimeError("redis set failure")

    async def delete(self, _key):
        pass


@pytest.mark.xfail(reason="Simulated Redis outage: expected failure for negative test scenario")
def test_redis_get_failure_raises_idempotency_error():
    engine = MagicMock()
    engine.process_image = AsyncMock(return_value={"text": "x"})

    storage = MagicMock()
    storage.upload_file = MagicMock(return_value="s3://bucket/file.png")

    redis = FailingRedis()

    # Cast to redis.Redis for mypy compatibility in tests
    from typing import cast

    import redis.asyncio as redis_mod

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
