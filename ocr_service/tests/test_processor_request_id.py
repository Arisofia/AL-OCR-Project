import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock

from fastapi import UploadFile

from ocr_service.modules.processor import OCRProcessor


def test_processor_returns_request_id():
    # Arrange: fake engine (async) and storage service (sync)
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
        filename = "test.png"
        content_type = "image/png"

        async def read(self):
            return b"image-bytes"

    # Act
    res = asyncio.run(
        processor.process_file(
            cast(UploadFile, DummyFile()),
            reconstruct=False,
            advanced=False,
            request_id="RID-123",
        )
    )

    # Assert
    assert "request_id" in res
    assert res["request_id"] == "RID-123"
