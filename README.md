# AL-OCR-Project
Sophisticated Fintech OCR System for financial document intelligence, built with FastAPI, React, and AWS (S3/Textract). Features automated infrastructure via Terraform and CI/CD.

## Testing & Local Development

This repository includes an automated script to create a reproducible Python 3.11 test environment and run the test-suite.

### Quick Start
1. Ensure Python 3.11 is available on your PATH.
2. Run the helper script:
   ```bash
   ./tools/setup_test_env.sh
   ```

The script will create a `.venv`, install dependencies, and run the tests.

### Fallback (Docker)
If Python 3.11 is not available locally, the script automatically falls back to running tests inside a Docker container.

### Continuous Integration
GitHub Actions workflows are configured to run tests on multiple Python versions (3.9, 3.10, 3.11) on each push/PR to `main`.

