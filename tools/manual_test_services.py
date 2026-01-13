from unittest.mock import MagicMock, patch
import sys

# Inject a fake boto3 module to avoid installation/typing issues in this environment
class FakeBoto3Module:
    def __init__(self):
        self._mocks = {}
    def client(self, name, *args, **kwargs):
        if name not in self._mocks:
            self._mocks[name] = MagicMock()
        return self._mocks[name]

sys.modules['boto3'] = FakeBoto3Module()

from services.storage import StorageService
from services.textract import TextractService

# Now run tests with the fake boto3
mock_s3 = sys.modules['boto3'].client('s3')
mock_tex = sys.modules['boto3'].client('textract')

# Test StorageService
s = StorageService(bucket_name='test-bucket')
key = s.upload_file(b'content', 'file.png', 'image/png')
print('upload_file returned key:', key)
saved = s.save_json({'a': 1}, 'out.json')
print('save_json returned:', saved)

# Test Textract
mock_tex.analyze_document.return_value = {'Blocks': []}
t = TextractService()
res = t.analyze_document('b', 'k')
print('analyze_document returned:', res)
