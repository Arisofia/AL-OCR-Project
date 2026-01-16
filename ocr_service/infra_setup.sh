#!/bin/bash
# Enterprise Infrastructure Provisioning Script for AL OCR Service
# Orchestrates baseline AWS resources including S3 and ECR with security best practices.

set -e

# --- Configuration & Defaults ---
AWS_REGION=${AWS_REGION:-"us-east-1"}

# Resolve Account ID
if [ -z "$AWS_ACCOUNT_ID" ]; then
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text 2>/dev/null)
fi

if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "Error: Could not resolve AWS Account ID. Please ensure you have AWS credentials configured."
    exit 1
fi

ECR_REPOSITORY=${ECR_REPOSITORY:-"al-ocr-service"}
S3_BUCKET_NAME=${S3_BUCKET_NAME:-"al-financial-documents-$AWS_ACCOUNT_ID"}

echo "Initializing infrastructure for Account: $AWS_ACCOUNT_ID in Region: $AWS_REGION"

# --- S3 Document Storage Layer ---
echo "Validating S3 Bucket: $S3_BUCKET_NAME..."
if aws s3api head-bucket --bucket "$S3_BUCKET_NAME" 2>/dev/null; then
  echo "Status: Bucket $S3_BUCKET_NAME already exists."
else
  echo "Action: Creating and hardening bucket $S3_BUCKET_NAME..."

  # Note: us-east-1 does not require LocationConstraint
  if [ "$AWS_REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket $S3_BUCKET_NAME --region $AWS_REGION
  else
    aws s3api create-bucket --bucket $S3_BUCKET_NAME --region $AWS_REGION \
      --create-bucket-configuration LocationConstraint=$AWS_REGION
  fi

  aws s3api put-public-access-block --bucket $S3_BUCKET_NAME --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  aws s3api put-bucket-encryption --bucket $S3_BUCKET_NAME --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  aws s3api put-bucket-versioning --bucket $S3_BUCKET_NAME --versioning-configuration Status=Enabled
  echo "Status: S3 Bucket configured with versioning and AES256 encryption."
fi

# --- ECR Container Registry Layer ---
echo "Validating ECR Repository: $ECR_REPOSITORY..."
if aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" --region $AWS_REGION >/dev/null 2>&1; then
  echo "Status: ECR repository $ECR_REPOSITORY already exists."
else
  echo "Action: Creating ECR repository $ECR_REPOSITORY..."
  aws ecr create-repository \
    --repository-name $ECR_REPOSITORY \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256
  echo "Status: ECR repository created with active image scanning."
fi

echo "Infrastructure initialization complete."
echo "--------------------------------------"
echo "Bucket Name: $S3_BUCKET_NAME"
echo "ECR Repo   : $ECR_REPOSITORY"
