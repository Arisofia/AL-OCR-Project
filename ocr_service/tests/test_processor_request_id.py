import asyncio
from unittest.mock import MagicMock
from typing import cast

from fastapi import UploadFile
from ocr_service.modules.processor import OCRProcessor


def test_processor_returns_request_id():
    # Arrange: fake engine (sync) and storage service (sync)
    engine = MagicMock()

    def fake_process_image(*_args, **_kwargs):
        return {"text": "dummy", "reconstruction": {}}

    engine.process_image = fake_process_image

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
        processor.process(
            cast(UploadFile, DummyFile()),
            reconstruct=False,
            advanced=False,
            request_id="RID-123",
        )
    )

    # Assert
    assert "request_id" in res
    assert res["request_id"] == "RID-123"
