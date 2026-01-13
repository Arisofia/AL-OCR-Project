provider "aws" {
  region = var.aws_region
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
  account_id         = var.account_id
  s3_bucket_name     = var.s3_bucket_name
  ecr_repository_name = var.ecr_repository_name
}
