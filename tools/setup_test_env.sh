#!/usr/bin/env bash
set -euo pipefail


# Setup script to create a reproducible test environment and run pytest.
# - Prefer local Python 3.11 if available
# - Fallback to Docker if Python 3.11 isn't available
# - Auto-export AWS_ACCOUNT_ID if not set (for AWS integration tests)
if [ -z "${AWS_ACCOUNT_ID:-}" ]; then
  export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "")
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_DIR="$PROJECT_ROOT/ocr_service"
VENV_DIR="$PROJECT_ROOT/.venv"
PYTHON_BIN=""

# Helper: check for python3.11
if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN=python3.11
elif command -v python3 >/dev/null 2>&1 && python3 --version 2>&1 | grep -q "3.11"; then
  PYTHON_BIN=python3
fi

if [ -n "${PYTHON_BIN}" ]; then
  echo "Using ${PYTHON_BIN} to create and populate venv at ${VENV_DIR}"
  if [ -d "${VENV_DIR}" ]; then
    echo "Removing existing venv at ${VENV_DIR} to avoid cross-version contamination"
    rm -rf "${VENV_DIR}"
  fi
  ${PYTHON_BIN} -m venv "${VENV_DIR}"
  PY_BIN="${VENV_DIR}/bin/python"
  PIP_BIN="${VENV_DIR}/bin/pip"
  "${PIP_BIN}" install --upgrade pip
  "${PIP_BIN}" install -r "${PROJECT_ROOT}/requirements.txt"
  "${PIP_BIN}" install -e "${PROJECT_ROOT}/ocr_reconstruct"
  "${PIP_BIN}" install pytest
  echo "Running pytest (local venv) from ${SERVICE_DIR}"
  pushd "${SERVICE_DIR}" >/dev/null
  PYTHONPATH="${PROJECT_ROOT}:${SERVICE_DIR}" "${PY_BIN}" -m pytest -q
  popd >/dev/null
else
  echo "Python 3.11 not found. Trying Docker fallback."
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker not available; please install Python 3.11 locally or Docker." >&2
    exit 2
  fi

  docker run --rm -v "${PROJECT_ROOT}":/app -w /app python:3.11 bash -lc "python -m pip install -U pip && python -m pip install -r requirements.txt pytest && python -m pip install -e ocr_reconstruct && PYTHONPATH=/app/ocr_service:/app python -m pytest -q"
fi
