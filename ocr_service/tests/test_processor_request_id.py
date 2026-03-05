"""Request ID propagation tests for OCR processor."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock

from fastapi import UploadFile

from ocr_service.modules.processor import OCRProcessor, ProcessingConfig


def test_processor_returns_request_id():
    engine = MagicMock()
    engine.process_image = AsyncMock(
        return_value={"text": "dummy", "reconstruction": {}}
    )
    engine.process_image_advanced = AsyncMock(
        return_value={"text": "dummy", "reconstruction": {}}
    )

    storage = MagicMock()

    def fake_upload_file(_content, filename, _content_type):
        return f"s3://bucket/{filename}"

    def fake_upload_json(_data, filename):
        return f"s3://bucket/{filename}.json"

    storage.upload_file = fake_upload_file
    storage.upload_json = fake_upload_json

    processor = OCRProcessor(engine, storage)

    class DummyFile:
        """Minimal UploadFile-compatible test double."""

        filename = "test.png"
        content_type = "image/png"

        async def read(self):
            await asyncio.sleep(0)
            return b"image-bytes"

    config = ProcessingConfig(
        reconstruct=False,
        advanced=False,
        request_id="RID-123",
    )
    res = asyncio.run(
        processor.process_file(
            cast(UploadFile, DummyFile()),
            config=config,
        )
    )

    assert "request_id" in res
    assert res["request_id"] == "RID-123"
