#!/usr/bin/env bash
# Enterprise Deployment Orchestrator for AL OCR Service
# Standardizes containerized deployments with automated ECR synchronization and Lambda updates.

set -euo pipefail

# --- Configuration & Identity Management ---
AWS_REGION=${AWS_REGION:?AWS_REGION is required}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID is required}
ECR_REPOSITORY=${ECR_REPOSITORY:-al-ocr-service}
DEFAULT_ECR_REPOSITORY="al-ocr-service"
LAMBDA_FUNCTION_NAME=${LAMBDA_FUNCTION_NAME:-AL-OCR-Processor}
AWS_LAMBDA_ROLE_ARN=${AWS_LAMBDA_ROLE_ARN:-}
LAMBDA_TIMEOUT=${LAMBDA_TIMEOUT:-30}
LAMBDA_MEMORY_SIZE=${LAMBDA_MEMORY_SIZE:-512}
LAMBDA_ARCHITECTURES=${LAMBDA_ARCHITECTURES:-x86_64}
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Normalize common secret input variants (e.g., full Lambda ARN with optional alias)
# and ensure the final name is valid for create-function.
if [[ "$LAMBDA_FUNCTION_NAME" == arn:*:function:* ]]; then
  LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME##*:function:}"
  LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME%%:*}"
fi

if [[ ! "$LAMBDA_FUNCTION_NAME" =~ ^[A-Za-z0-9_-]{1,64}$ ]]; then
  echo "Invalid LAMBDA_FUNCTION_NAME value detected. Falling back to AL-OCR-Processor."
  LAMBDA_FUNCTION_NAME="AL-OCR-Processor"
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
echo "Updating Lambda function: $LAMBDA_FUNCTION_NAME"
if aws lambda get-function --function-name "$LAMBDA_FUNCTION_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  aws lambda update-function-code \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --image-uri "$IMAGE_URI" \
    --region "$AWS_REGION"
else
  if [[ -z "$AWS_LAMBDA_ROLE_ARN" ]]; then
    echo "Error: Lambda function $LAMBDA_FUNCTION_NAME does not exist and AWS_LAMBDA_ROLE_ARN is not set."
    echo "Set AWS_LAMBDA_ROLE_ARN (GitHub Actions secret) to allow automatic function creation."
    exit 1
  fi

  echo "Lambda function $LAMBDA_FUNCTION_NAME does not exist. Creating it now."
  aws lambda create-function \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --package-type Image \
    --code ImageUri="$IMAGE_URI" \
    --role "$AWS_LAMBDA_ROLE_ARN" \
    --timeout "$LAMBDA_TIMEOUT" \
    --memory-size "$LAMBDA_MEMORY_SIZE" \
    --architectures "$LAMBDA_ARCHITECTURES" \
    --region "$AWS_REGION" >/dev/null
  echo "Created Lambda function: $LAMBDA_FUNCTION_NAME"
fi

echo "Deployment completed successfully. Version: $COMMIT_TAG"
