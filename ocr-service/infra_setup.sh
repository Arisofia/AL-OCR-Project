#!/bin/bash

# Configuration
AWS_REGION="us-east-1"
ECR_REPOSITORY="al-ocr-service"
S3_BUCKET_NAME="al-financial-documents-510701314494" # Unique bucket name

echo "Creating S3 Bucket: $S3_BUCKET_NAME..."
# Create bucket if it doesn't exist
if aws s3api head-bucket --bucket "$S3_BUCKET_NAME" 2>/dev/null; then
  echo "Bucket $S3_BUCKET_NAME already exists. Skipping creation."
else
  aws s3api create-bucket --bucket $S3_BUCKET_NAME --region $AWS_REGION \
    --create-bucket-configuration LocationConstraint=$AWS_REGION
  aws s3api put-public-access-block --bucket $S3_BUCKET_NAME --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  aws s3api put-bucket-encryption --bucket $S3_BUCKET_NAME --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  aws s3api put-bucket-versioning --bucket $S3_BUCKET_NAME --versioning-configuration Status=Enabled
  echo "Created and configured bucket $S3_BUCKET_NAME"
fi

echo "Creating ECR Repository: $ECR_REPOSITORY..."
# Create repo if it doesn't exist
if aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" --region $AWS_REGION >/dev/null 2>&1; then
  echo "ECR repository $ECR_REPOSITORY already exists. Skipping creation."
else
  aws ecr create-repository \
    --repository-name $ECR_REPOSITORY \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256
  echo "Created ECR repository $ECR_REPOSITORY"
fi

echo "Infrastructure components created successfully."
echo "Bucket Name: $S3_BUCKET_NAME"
echo "ECR Repo: $ECR_REPOSITORY"
