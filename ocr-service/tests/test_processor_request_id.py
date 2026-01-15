import asyncio
from unittest.mock import MagicMock

from modules.processor import OCRProcessor


def test_processor_returns_request_id():
    # Arrange: fake engine (sync) and storage service (sync)
    engine = MagicMock()

    def fake_process_image(contents, use_reconstruction=False):
        return {"text": "dummy", "reconstruction": {}}

    engine.process_image = fake_process_image

    storage = MagicMock()
    storage.upload_file = lambda content, filename, content_type: f"s3://bucket/{filename}"
    storage.upload_json = lambda data, filename: f"s3://bucket/{filename}.json"

    processor = OCRProcessor(engine, storage)

    class DummyFile:
        filename = "test.png"
        content_type = "image/png"

        async def read(self):
            return b"image-bytes"

    # Act
    res = asyncio.run(
        processor.process(
            DummyFile(), reconstruct=False, advanced=False, request_id="RID-123"
        )
    )

    # Assert
    assert "request_id" in res
    assert res["request_id"] == "RID-123"
