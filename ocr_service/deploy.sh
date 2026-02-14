#!/usr/bin/env bash
# Enterprise Deployment Orchestrator for AL OCR Service
# Standardizes containerized deployments with automated ECR synchronization and Lambda updates.

set -euo pipefail

# --- Configuration & Identity Management ---
AWS_REGION=${AWS_REGION:?AWS_REGION is required}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID is required}
ECR_REPOSITORY=${ECR_REPOSITORY:-al-ocr-service}
DEFAULT_ECR_REPOSITORY="al-ocr-service"
DEFAULT_LAMBDA_FUNCTION_NAME="AL-OCR-Upload-API"
LAMBDA_FUNCTION_NAME=${LAMBDA_FUNCTION_NAME:-$DEFAULT_LAMBDA_FUNCTION_NAME}
AWS_LAMBDA_ROLE_ARN=${AWS_LAMBDA_ROLE_ARN:-}
LAMBDA_TIMEOUT=${LAMBDA_TIMEOUT:-30}
LAMBDA_MEMORY_SIZE=${LAMBDA_MEMORY_SIZE:-512}
LAMBDA_ARCHITECTURES=${LAMBDA_ARCHITECTURES:-x86_64}
LAMBDA_DEPLOY_STRICT=${LAMBDA_DEPLOY_STRICT:-true}
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SKIP_LAMBDA_UPDATE=0

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

warn_or_fail_lambda() {
  local message="$1"
  if is_truthy "$LAMBDA_DEPLOY_STRICT"; then
    echo "Error: $message"
    exit 1
  fi
  echo "Warning: $message"
  echo "Continuing because LAMBDA_DEPLOY_STRICT=$LAMBDA_DEPLOY_STRICT"
}

normalize_lambda_function_name() {
  local raw="$1"

  raw="${raw//$'\r'/}"
  raw="${raw#"${raw%%[![:space:]]*}"}"
  raw="${raw%"${raw##*[![:space:]]}"}"

  if [[ "$raw" == \"*\" ]]; then
    raw="${raw#\"}"
    raw="${raw%\"}"
  elif [[ "$raw" == \'*\' ]]; then
    raw="${raw#\'}"
    raw="${raw%\'}"
  fi

  if [[ "$raw" == arn:*:function:* ]]; then
    raw="${raw##*:function:}"
  fi

  # Function qualifiers can be provided as "<name>:<alias>".
  raw="${raw%%:*}"
  echo "$raw"
}

# Normalize common secret input variants (e.g., full Lambda ARN with optional alias)
# and ensure the final name is valid for create-function.
LAMBDA_FUNCTION_NAME="$(normalize_lambda_function_name "$LAMBDA_FUNCTION_NAME")"

if [[ ! "$LAMBDA_FUNCTION_NAME" =~ ^[A-Za-z0-9_-]{1,64}$ ]]; then
  warn_or_fail_lambda "Invalid LAMBDA_FUNCTION_NAME value detected. Set AWS_LAMBDA_FUNCTION_NAME to a valid Lambda name."
  LAMBDA_FUNCTION_NAME="$DEFAULT_LAMBDA_FUNCTION_NAME"
fi

if [[ ! "$LAMBDA_FUNCTION_NAME" =~ ^[A-Za-z0-9_-]{1,64}$ ]]; then
  warn_or_fail_lambda "Unable to recover a valid Lambda name. Skipping Lambda update."
  SKIP_LAMBDA_UPDATE=1
fi

echo "Initializing deployment for account: $AWS_ACCOUNT_ID in region: $AWS_REGION"

# --- Authentication & Registry Access ---
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

# --- Build & Artifact Versioning ---
COMMIT_TAG=$(git rev-parse --short HEAD || echo "local-$(date +%s)")

# --- Infrastructure Validation ---
if ! aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "ECR repository $ECR_REPOSITORY does not exist. Creating it now."
  if aws ecr create-repository \
    --repository-name "$ECR_REPOSITORY" \
    --region "$AWS_REGION" \
    --image-scanning-configuration scanOnPush=true \
    --image-tag-mutability MUTABLE >/dev/null 2>&1; then
    echo "Created ECR repository: $ECR_REPOSITORY"
  elif [[ "$ECR_REPOSITORY" != "$DEFAULT_ECR_REPOSITORY" ]] && \
       aws ecr describe-repositories --repository-names "$DEFAULT_ECR_REPOSITORY" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo "Insufficient permission to create $ECR_REPOSITORY. Falling back to existing repository $DEFAULT_ECR_REPOSITORY."
    ECR_REPOSITORY="$DEFAULT_ECR_REPOSITORY"
  else
    echo "Error: Unable to create ECR repository $ECR_REPOSITORY."
    echo "Grant ecr:CreateRepository to the deployment role or set ECR_REPOSITORY to an existing repository."
    exit 1
  fi
fi

IMAGE_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$COMMIT_TAG

# --- Container Image Construction ---
echo "Building container image: $IMAGE_URI"
docker buildx create --use || true
LATEST_TAG="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest"
docker buildx build \
  --platform linux/amd64 \
  -t "$IMAGE_URI" \
  -t "$LATEST_TAG" \
  --push \
  -f "$REPO_ROOT/ocr_service/Dockerfile" \
  "$REPO_ROOT"

# --- Lambda Lifecycle Update ---
if [[ "$SKIP_LAMBDA_UPDATE" -eq 0 ]]; then
  echo "Updating Lambda function: $LAMBDA_FUNCTION_NAME"
  if aws lambda get-function --function-name "$LAMBDA_FUNCTION_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
    if ! aws lambda update-function-code \
      --function-name "$LAMBDA_FUNCTION_NAME" \
      --image-uri "$IMAGE_URI" \
      --region "$AWS_REGION"; then
      warn_or_fail_lambda "Unable to update Lambda function $LAMBDA_FUNCTION_NAME."
    fi
  else
    if [[ -z "$AWS_LAMBDA_ROLE_ARN" ]]; then
      warn_or_fail_lambda "Lambda function $LAMBDA_FUNCTION_NAME does not exist and AWS_LAMBDA_ROLE_ARN is not set."
    else
      echo "Lambda function $LAMBDA_FUNCTION_NAME does not exist. Creating it now."
      create_err_file="$(mktemp)"
      if aws lambda create-function \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --package-type Image \
        --code ImageUri="$IMAGE_URI" \
        --role "$AWS_LAMBDA_ROLE_ARN" \
        --timeout "$LAMBDA_TIMEOUT" \
        --memory-size "$LAMBDA_MEMORY_SIZE" \
        --architectures "$LAMBDA_ARCHITECTURES" \
        --region "$AWS_REGION" >/dev/null 2>"$create_err_file"; then
        echo "Created Lambda function: $LAMBDA_FUNCTION_NAME"
      else
        create_error="$(tr '\n' ' ' < "$create_err_file" | sed 's/[[:space:]]\+/ /g')"
        rm -f "$create_err_file"
        warn_or_fail_lambda "Unable to create Lambda function $LAMBDA_FUNCTION_NAME. ${create_error}"
      fi
      rm -f "$create_err_file"
    fi
  fi
else
  echo "Skipping Lambda update due to invalid Lambda configuration."
fi

echo "Deployment completed successfully. Version: $COMMIT_TAG"
