variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "account_id" {
  description = "AWS Account ID (defaults to current caller if empty)"
  type        = string
  default     = ""
}

variable "s3_bucket_name" {
  description = "Name of the S3 bucket for documents (defaults to al-financial-documents-<ACCOUNT_ID> if left as placeholder)"
  type        = string
  default     = "al-financial-documents-PLACEHOLDER"
}

variable "ecr_repository_name" {
  description = "Name of the ECR repository"
  type        = string
  default     = "al-ocr-service"
}

variable "ecr_force_delete" {
  description = "When true, allow Terraform to delete images by setting force_delete true on ECR repo (destructive)"
  type        = bool
  default     = false
}

variable "ecr_image_tag_mutability" {
  description = "ECR image tag mutability: IMMUTABLE or MUTABLE"
  type        = string
  default     = "IMMUTABLE"
}
