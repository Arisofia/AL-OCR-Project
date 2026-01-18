from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError

from ocr_service.services.worker import WorkerService


@pytest.fixture
def worker():
    return WorkerService()


@pytest.fixture
def s3_record():
    return {
        "s3": {
            "bucket": {"name": "test-bucket"},
            "object": {"key": "test+file.jpg"},
        }
    }


def test_process_s3_record_success_image(worker, s3_record):
    with (
        patch("ocr_service.services.worker.StorageService") as mock_storage_cls,
        patch.object(worker.textract_service, "analyze_document") as mock_analyze,
    ):
        mock_storage = mock_storage_cls.return_value
        mock_analyze.return_value = {"text": "found it"}
        mock_storage.save_json.return_value = True

        worker.process_s3_record(s3_record, request_id="RID-123")

        mock_analyze.assert_called_once_with("test-bucket", "test file.jpg")
        mock_storage.save_json.assert_called_once()
        args, _ = mock_storage.save_json.call_args
        assert args[0]["text"] == "found it"
        assert args[0]["requestId"] == "RID-123"


def test_process_s3_record_success_pdf(worker):
    pdf_record = {
        "s3": {
            "bucket": {"name": "test-bucket"},
            "object": {"key": "test.pdf"},
        }
    }
    with (
        patch("ocr_service.services.worker.StorageService") as mock_storage_cls,
        patch.object(worker.textract_service, "start_detection") as mock_start,
    ):
        mock_storage = mock_storage_cls.return_value
        mock_start.return_value = "job-123"
        mock_storage.save_json.return_value = True

        worker.process_s3_record(pdf_record, request_id="RID-123")

        mock_start.assert_called_once_with("test-bucket", "test.pdf")
        mock_storage.save_json.assert_called_once()
        args, _ = mock_storage.save_json.call_args
        assert args[0]["jobId"] == "job-123"
        assert args[0]["status"] == "STARTED"


def test_process_s3_record_aws_error(worker, s3_record):
    with (
        patch("ocr_service.services.worker.StorageService") as mock_storage_cls,
        patch.object(worker.textract_service, "analyze_document") as mock_analyze,
    ):
        mock_storage = mock_storage_cls.return_value
        error_response = {
            "Error": {"Code": "AccessDenied", "Message": "No access"},
            "ResponseMetadata": {"RequestId": "AWS-RID-999"},
        }
        mock_analyze.side_effect = ClientError(error_response, "AnalyzeDocument")

        with pytest.raises(ClientError):
            worker.process_s3_record(s3_record)

        mock_storage.save_json.assert_called_once()
        args, _ = mock_storage.save_json.call_args
        assert args[0]["error"] == "aws_service_failure"
        assert args[0]["requestId"] == "AWS-RID-999"
