"""
Manual test script for verifying AWS services integration.
Uses mocking to simulate AWS interactions without requiring real credentials.
"""

import os
import sys
import typing
from unittest.mock import MagicMock

# Add ocr_service to path
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ocr_service"))
except Exception:
    pass


class FakeBoto3Module:
    """
    Fake boto3 module to avoid installation/typing issues in this environment.
    """

    def __init__(self):
        self._mocks = {}

    def client(self, name, *args, **kwargs):  # pylint: disable=unused-argument
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

# Inject fake boto3 BEFORE importing services
# Service imports are performed inside `run_tests` after the fake boto3 module
# is registered to avoid top-level import-time side effects and E402 lint warnings.


def run_tests():
    """
    Executes a basic set of tests for storage and textract services.
    """
    # Now run tests with the fake boto3
    mock_tex = sys.modules["boto3"].client("textract")

    # Import services here after faking boto3 to avoid import-time side effects
    from services.storage import StorageService
    from services.textract import TextractService

    # Test StorageService
    storage = StorageService(bucket_name="test-bucket")
    key = storage.upload_file(b"content", "file.png", "image/png")
    print("upload_file returned key:", key)
    saved = storage.save_json({"a": 1}, "out.json")
    print("save_json returned:", saved)

    # Test Textract
    mock_tex.analyze_document.return_value = {"Blocks": []}
    textract = TextractService()
    res = textract.analyze_document(b"b", "k")
    print("analyze_document returned:", res)


if __name__ == "__main__":
    run_tests()
