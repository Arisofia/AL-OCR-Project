variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "account_id" {
  description = "AWS account ID"
  type        = string
}

variable "s3_bucket_name" {
  description = "S3 bucket name for documents"
  type        = string
}

variable "lambda_function_name" {
  description = "Lambda function name for OCR processor"
  type        = string
  default     = "al-ocr-processor"
}

variable "image_tag" {
  description = "ECR image tag to deploy (e.g., commit SHA or 'latest')"
  type        = string
  default     = "latest"
}