Testing & local dev environment

This repository includes an automated script to create a reproducible Python 3.11 test environment and run the test-suite.

Quick start (preferred: local Python 3.11):

1. Ensure Python 3.11 is available on your PATH (install via pyenv or system package manager).
2. Run the helper script:

   ./tools/setup_test_env.sh

The script will create a `.venv` in the repo root, install dependencies from `ocr-service/requirements.txt`, install pytest, and run the tests with `PYTHONPATH=ocr-service`.

Fallback (Docker):

If Python 3.11 is not available, the script will automatically fall back to running tests inside a Python 3.11 Docker container.

CI:

A GitHub Actions workflow (`.github/workflows/ci-python.yml`) is included to run tests on Python 3.11 on each push/PR to `main`.

Notes:
- If you prefer a different Python version for local dev, update the script accordingly, but tests are verified against Python 3.11 in CI.
- If you encounter typing/import errors stemming from a third-party `typing` package, removing that package from the virtual environment (`pip uninstall typing`) after switching to Python 3.11 is recommended.
