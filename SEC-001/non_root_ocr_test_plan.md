# Test Plan: Security Hardening (Non-root User Implementation)

## Objectives:
- Validate that the `ocr_service` Docker image builds successfully using a non-root user.
- Verify that the service runs correctly under the `appuser` identity without permission issues.
- Ensure that OCR processing and file system operations function as expected in a restricted environment.
- Confirm compliance with security best practices for containerized applications.

## Scope:
- Dockerfile build process for `ocr_service`.
- Runtime execution of the FastAPI/Lambda handler.
- Filesystem permissions for `/var/task` and `/tmp`.
- Integration with Tesseract and OpenCV under non-root context.

## Out of Scope:
- Performance benchmarking of the OCR engine itself.
- Security of the AWS infrastructure (IAM, S3 buckets) outside the container context.

## Test Approach:
- **Static Analysis**: Verify Dockerfile instructions for best practices.
- **Build Verification**: Ensure `shadow-utils` and `useradd` work correctly during image creation.
- **Functional Testing**: Run the container locally and invoke the handler to verify end-to-end OCR logic.
- **Security Testing**: Execute `whoami` and `id` inside the container to confirm non-root execution.
- **Regression Testing**: Ensure existing OCR capabilities (S3 triggers, Textract integration) are not broken.

## Test Environment Requirements:
- Docker Engine.
- AWS Credentials (mocked or real for S3/Textract integration).
- LocalStack (optional) for local cloud simulation.

## Risk Assessment:
- **Permission Denied**: `appuser` might not have write access to required directories (e.g., `/tmp` or specific subfolders).
- **Broken Dependencies**: System libraries (OpenCV, Tesseract) might expect root privileges for certain operations.
- **Build Failure**: Incompatibility with the base Lambda image (Amazon Linux 2) regarding user management tools.

## Key Checklist Items:
- [ ] Image builds without `useradd: command not found` error.
- [ ] `whoami` returns `appuser`.
- [ ] File ownership of `/var/task` is `appuser:appuser`.
- [ ] OCR processing completes successfully without permission errors.

## Test Exit Criteria:
- Docker image builds and starts successfully.
- 100% of Critical and High priority test cases pass.
- No "Permission Denied" errors logged during standard OCR workflows.
