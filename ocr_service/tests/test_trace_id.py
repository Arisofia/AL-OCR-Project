import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
import redis.asyncio as redis

from ocr_service.exceptions import OCRPipelineError
from ocr_service.modules.processor import OCRProcessor, ProcessingConfig


class FakeRedis:
    def __init__(self):
        self.store = {}

    async def get(self, _key):
        return None

    async def set(self, _key, _value, _ex=None):
        self.store[_key] = _value

    async def delete(self, _key):
        self.store.pop(_key, None)


@pytest.mark.parametrize(
    "trace_val,expected",
    [
        (None, None),
        (0x1A2B3C, format(0x1A2B3C, "x")),
    ],
)
def test_trace_id_propagation_on_extraction_error(monkeypatch, trace_val, expected):
    """When extraction fails, the raised OCRPipelineError should carry the
    optional `trace_id`.

    - If the span context has no trace id, `exception.trace_id` should be None.
    - If the span context has an integer trace id, it should be formatted as
      a hex string.
    """
    # Arrange
    engine = MagicMock()
    engine.process_image = AsyncMock(return_value={"error": "boom"})
    engine.process_image_advanced = AsyncMock(return_value={"error": "boom"})

    storage = MagicMock()
    fake_redis = FakeRedis()

    span = MagicMock()
    span.get_span_context = MagicMock(return_value=MagicMock(trace_id=trace_val))
    monkeypatch.setattr("opentelemetry.trace.get_current_span", lambda: span)

    processor = OCRProcessor(engine, storage, cast(redis.Redis, fake_redis))

    # Act / Assert
    with pytest.raises(OCRPipelineError) as exc:
        asyncio.run(
            processor.process_bytes(
                contents=b"img",
                filename="file.png",
                content_type="image/png",
                config=ProcessingConfig(),
            )
        )

    assert exc.value.trace_id == expected
