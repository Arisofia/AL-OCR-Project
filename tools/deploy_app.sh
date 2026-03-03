#!/usr/bin/env bash
set -euo pipefail

# Explicitly require deployment credentials/inputs passed by CI.
: "${API_KEY:?API_KEY is required}"
: "${ACCESS_TOKEN:?ACCESS_TOKEN is required}"
: "${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID is required}"
: "${AWS_REGION:?AWS_REGION is required}"

# Expose OCR_API_KEY expected by backend deployment scripts.
export OCR_API_KEY="${OCR_API_KEY:-$API_KEY}"

bash ocr_service/deploy.sh
