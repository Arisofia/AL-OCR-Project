"""
Manual test script for verifying AWS services integration.
Uses mocking to simulate AWS interactions without requiring real credentials.
"""

import contextlib
import importlib
import sys
import typing
from pathlib import Path
from unittest.mock import MagicMock

with contextlib.suppress(Exception):
    PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)


class FakeBoto3Module:
    """
    Fake boto3 module to avoid installation/typing issues in this environment.
    """

    def __init__(self):
        self._mocks = {}

    def client(self, name, *args, **kwargs):
        """
        Returns a mocked client.
        """
        if name not in self._mocks:
            self._mocks[name] = MagicMock()
        return self._mocks[name]


sys.modules["boto3"] = typing.cast(typing.Any, FakeBoto3Module())
sys.modules["botocore"] = MagicMock()
sys.modules["botocore.exceptions"] = MagicMock()
sys.modules["botocore.config"] = MagicMock()

StorageService = importlib.import_module("ocr_service.services.storage").StorageService
TextractService = importlib.import_module(
    "ocr_service.services.textract"
).TextractService


def run_tests():
    """
    Executes a basic set of tests for storage and textract services.
    """
    mock_tex = sys.modules["boto3"].client("textract")

    storage = StorageService(bucket_name="test-bucket")
    key = storage.upload_file(b"content", "file.png", "image/png")
    print("upload_file returned key:", key)
    saved = storage.save_json({"a": 1}, "out.json")
    print("save_json returned:", saved)

    mock_tex.analyze_document.return_value = {"Blocks": []}
    textract = TextractService()
    res = textract.analyze_document("test-bucket", "test-key")
    print("analyze_document returned:", res)


if __name__ == "__main__":
    run_tests()
