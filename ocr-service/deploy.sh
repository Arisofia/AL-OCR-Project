#!/bin/bash

# Configuration
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="510701314494"
ECR_REPOSITORY="al-ocr-service"
LAMBDA_FUNCTION_NAME="AL-OCR-Processor"

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build and push using buildx with commit SHA tag
COMMIT_TAG=$(git rev-parse --short HEAD || echo "local")
IMAGE_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$COMMIT_TAG

# Ensure repository exists
if ! aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" --region $AWS_REGION >/dev/null 2>&1; then
  echo "ECR repo $ECR_REPOSITORY does not exist. Run infra_setup.sh first."
  exit 1
fi

# Build and push image
docker buildx build --platform linux/amd64 -t $IMAGE_URI --push .

# Also push latest tag
docker tag $IMAGE_URI $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest

# Update Lambda function code
aws lambda update-function-code \
    --function-name $LAMBDA_FUNCTION_NAME \
    --image-uri $IMAGE_URI

if [ $? -eq 0 ]; then
  echo "Lambda updated successfully with image $IMAGE_URI"
else
  echo "Failed to update Lambda."
  exit 1
fi
