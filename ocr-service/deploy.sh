#!/bin/bash
# Enterprise Deployment Orchestrator for AL OCR Service
# Standardizes containerized deployments with automated ECR synchronization and Lambda updates.

set -e # Exit immediately on error

# --- Configuration & Identity Management ---
AWS_REGION=${AWS_REGION:-"us-east-1"}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-"510701314494"}
ECR_REPOSITORY=${ECR_REPOSITORY:-"al-ocr-service"}
LAMBDA_FUNCTION_NAME=${LAMBDA_FUNCTION_NAME:-"AL-OCR-Processor"}

echo "Initializing deployment for account: $AWS_ACCOUNT_ID in region: $AWS_REGION"

# --- Authentication & Registry Access ---
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# --- Build & Artifact Versioning ---
COMMIT_TAG=$(git rev-parse --short HEAD || echo "local-$(date +%s)")
IMAGE_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$COMMIT_TAG

# --- Infrastructure Validation ---
if ! aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" --region $AWS_REGION >/dev/null 2>&1; then
  echo "Error: ECR repository $ECR_REPOSITORY does not exist. Ensure infrastructure is provisioned via Terraform."
  exit 1
fi

# --- Container Image Construction ---
echo "Building container image: $IMAGE_URI"
docker buildx build --platform linux/amd64 -t $IMAGE_URI --push .

# --- Alias Management (Latest) ---
LATEST_TAG="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest"
docker tag $IMAGE_URI $LATEST_TAG
docker push $LATEST_TAG

# --- Lambda Lifecycle Update ---
echo "Updating Lambda function: $LAMBDA_FUNCTION_NAME"
aws lambda update-function-code \
    --function-name $LAMBDA_FUNCTION_NAME \
    --image-uri $IMAGE_URI

echo "Deployment completed successfully. Version: $COMMIT_TAG"
