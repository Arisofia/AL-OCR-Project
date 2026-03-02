# Detailed Test Cases: Non-root OCR Service

## Test Case ID: TC-SEC-01
- **Test Case Title**: Verify Docker Image Build Success
- **Priority**: Critical
- **Type**: Functional
- **Preconditions**: Docker installed, `shadow-utils` included in Dockerfile.
- **Tags**: #build #security
- **Test Data Requirements**: None
- **Parameters**: None
- **Test Steps - Data - Expected Result**
1. Run `docker build -t ocr-service:test -f ocr_service/Dockerfile .` - N/A - Image builds successfully without `command not found` errors.

---

## Test Case ID: TC-SEC-02
- **Test Case Title**: Confirm Identity of Running User
- **Priority**: Critical
- **Type**: Security
- **Preconditions**: Image built successfully.
- **Tags**: #security #runtime
- **Test Data Requirements**: None
- **Parameters**: None
- **Test Steps - Data - Expected Result**
1. Run `docker run --entrypoint whoami ocr-service:test` - N/A - Output is `appuser`.
2. Run `docker run --entrypoint id ocr-service:test` - N/A - Output shows UID/GID for `appuser` (usually 1000 or similar).

---

## Test Case ID: TC-SEC-03
- **Test Case Title**: Verify /var/task Ownership and Permissions
- **Priority**: High
- **Type**: Security
- **Preconditions**: Container running.
- **Tags**: #security #permissions
- **Test Data Requirements**: None
- **Parameters**: None
- **Test Steps - Data - Expected Result**
1. Run `ls -ld /var/task` - N/A - Ownership is `appuser:appuser`.
2. Attempt to create a file in `/var/task` - `touch /var/task/test.txt` - File is created successfully (if permissions allow) or fails as expected if read-only, but ownership must be correct.

---

## Test Case ID: TC-SEC-04
- **Test Case Title**: Verify Write Permissions to /tmp
- **Priority**: High
- **Type**: Functional
- **Preconditions**: Container running as `appuser`.
- **Tags**: #functional #lambda
- **Test Data Requirements**: None
- **Parameters**: None
- **Test Steps - Data - Expected Result**
1. Execute `touch /tmp/test_write` - N/A - File created successfully without "Permission denied".
2. Execute `rm /tmp/test_write` - N/A - File deleted successfully.

---

## Test Case ID: TC-SEC-05
- **Test Case Title**: End-to-End OCR Processing as Non-root
- **Priority**: Critical
- **Type**: Functional
- **Preconditions**: Valid S3 event payload.
- **Tags**: #functional #ocr
- **Test Data Requirements**: Sample PDF/Image in S3 (or mocked).
- **Parameters**: `image_path`
- **Test Steps - Data - Expected Result**
1. Invoke Lambda handler with test event - `S3 Event` - OCR process completes, results are uploaded to S3, and no permission errors are logged.

---

## Test Case ID: TC-SEC-07
- **Test Case Title**: Verify Access to Tesseract System Data
- **Priority**: High
- **Type**: Functional
- **Preconditions**: Tesseract installed in image.
- **Tags**: #functional #tesseract
- **Test Data Requirements**: None
- **Parameters**: None
- **Test Steps - Data - Expected Result**
1. Run `tesseract --list-langs` - N/A - Output lists `eng`, `spa`, etc., confirming `appuser` can read system-wide language data.
