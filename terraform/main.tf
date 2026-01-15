provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

locals {
  account_id = var.account_id != "" ? var.account_id : data.aws_caller_identity.current.account_id
}

terraform {
  required_version = ">= 1.0.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

module "ocr_infrastructure" {
  source = "./modules/ocr_service"

  aws_region         = var.aws_region
  account_id         = local.account_id
  s3_bucket_name     = var.s3_bucket_name == "al-financial-documents-PLACEHOLDER" ? "al-financial-documents-${local.account_id}" : var.s3_bucket_name
  ecr_repository_name = var.ecr_repository_name

  # Pass through ECR module configuration
  ecr_force_delete           = var.ecr_force_delete
  ecr_image_tag_mutability  = var.ecr_image_tag_mutability
}

output "github_actions_role_arn" {
  value = module.ocr_infrastructure.github_actions_role_arn
}

output "lambda_role_arn" {
  value = module.ocr_infrastructure.lambda_role_arn
}

output "s3_bucket_arn" {
  value = module.ocr_infrastructure.s3_bucket_arn
}

output "ecr_repository_url" {
  value = module.ocr_infrastructure.ecr_repository_url
}
