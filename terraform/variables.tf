variable "aws_region" {
  description = "AWS region"
  type        = "string"
  default     = "us-east-1"
}

variable "account_id" {
  description = "AWS Account ID"
  type        = "string"
  default     = "510701314494"
}

variable "s3_bucket_name" {
  description = "Name of the S3 bucket for documents"
  type        = "string"
  default     = "al-financial-documents-510701314494"
}

variable "ecr_repository_name" {
  description = "Name of the ECR repository"
  type        = "string"
  default     = "al-ocr-service"
}
