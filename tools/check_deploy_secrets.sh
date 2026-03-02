#!/usr/bin/env bash
set -euo pipefail

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

missing=0

check_var() {
  local name="$1"
  local required="$2"
  local value="${!name:-}"

  if [[ -n "$value" ]]; then
    echo "[secrets] OK: $name"
    return
  fi

  if [[ "$required" == "required" ]]; then
    echo "[secrets] MISSING (required): $name"
    missing=1
  else
    echo "[secrets] MISSING (optional): $name"
  fi
}

echo "[secrets] Checking deploy and CI/CD secret environment values"

# Primary deploy workflow (push to main + npm run deploy)
check_var OCR_API_KEY required
check_var AWS_ACCOUNT_ID required
check_var AWS_ROLE_TO_ASSUME required
check_var AWS_REGION required

# Optional deploy/build settings
check_var ECR_REPOSITORY optional
check_var AWS_LAMBDA_FUNCTION_NAME optional
check_var S3_BUCKET_NAME optional

# Legacy and extended workflows
check_var DEPLOY_HOST optional
check_var DEPLOY_USER optional
check_var DEPLOY_PORT optional
check_var DEPLOY_SSH_KEY optional
check_var REDIS_PASSWORD optional
check_var AWS_ACCESS_KEY_ID optional
check_var AWS_SECRET_ACCESS_KEY optional
check_var AWS_LAMBDA_ROLE_ARN optional
check_var ENABLE_LAMBDA_ROLLBACK optional
check_var ROLLBACK_COLD_MS optional
check_var SLACK_WEBHOOK_URL optional

# E2E and external integrations
check_var STAGING_API_KEY optional
check_var NETLIFY_AUTH_TOKEN optional

if [[ $missing -ne 0 && $STRICT -eq 1 ]]; then
  echo "[secrets] Required secrets are missing and --strict is enabled."
  exit 1
fi

if [[ $missing -ne 0 ]]; then
  echo "[secrets] Required secrets are missing. Set them before production deployment."
else
  echo "[secrets] All required deploy secrets are present in environment."
fi
