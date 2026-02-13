# pylint: disable=protected-access

import asyncio
from unittest.mock import AsyncMock, MagicMock

from ocr_service import metrics
from ocr_service.exceptions import OCRPipelineError
from ocr_service.modules.processor import OCRProcessor, ProcessingConfig


class FailingRedis:
    async def get(self, _key):
        raise RuntimeError("redis connection failure")

    async def set(self, _key, _value, _ex=None):
        raise RuntimeError("redis set failure")

    async def delete(self, _key):
        raise RuntimeError("redis delete failure")


def test_metrics_increment_on_redis_get_failure():
    engine = MagicMock()
    engine.process_image = AsyncMock(return_value={"text": "x"})

    storage = MagicMock()
    storage.upload_file = MagicMock(return_value="s3://bucket/file.png")

    import contextlib
    from typing import cast

    import redis.asyncio as redis_mod

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
