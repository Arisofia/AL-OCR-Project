#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTEST_ARGS=()
if ! python3 - <<'PY' >/dev/null 2>&1
import pytest_cov  # noqa: F401
PY
then
  PYTEST_ARGS=(-o addopts='')
fi

echo "[validate] Running backend exception guard"
python3 -m ocr_service.scripts.check_exception_handlers

echo "[validate] Verifying CI/CD workflow configuration"
./tools/verify_cicd.sh

echo "[validate] Running backend tests"
python3 -m pytest "${PYTEST_ARGS[@]}"

echo "[validate] Running frontend lint"
npm run frontend:lint

echo "[validate] Running frontend production build"
npm run frontend:build

echo "[validate] Checking deploy secret configuration (non-strict)"
./tools/check_deploy_secrets.sh

echo "[validate] Repository validation complete"
