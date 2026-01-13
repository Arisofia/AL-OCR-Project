terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.0.0"
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "account_id" {
  type = string
}

variable "s3_bucket_name" {
  type    = string
}

resource "aws_s3_bucket" "documents" {
  bucket = var.s3_bucket_name

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  versioning {
    enabled = true
  }

  lifecycle_rule {
    id      = "expire-old-objects"
    enabled = true

    expiration {
      days = 365
    }

    noncurrent_version_expiration {
      days = 90
    }
  }

  tags = {
    Name        = var.s3_bucket_name
    Environment = "dev"
  }
}

resource "aws_ecr_repository" "ocr_service" {
  name                 = "al-ocr-service"
  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_iam_role" "lambda_exec" {
  name = "al_ocr_lambda_exec"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "al_ocr_lambda_policy"
  role = aws_iam_role.lambda_exec.id

  policy = data.aws_iam_policy_document.lambda_policy.json
}

data "aws_iam_policy_document" "lambda_policy" {
  statement {
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
      "textract:AnalyzeDocument",
      "textract:StartDocumentTextDetection",
      "textract:GetDocumentTextDetection",
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents"
    ]

    resources = ["*"]
  }
}

# Lambda function (packaged as an ECR image)
resource "aws_lambda_function" "ocr_processor" {
  function_name = var.lambda_function_name
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.ocr_service.repository_url}:${var.image_tag}"
  role          = aws_iam_role.lambda_exec.arn
  timeout       = 60
  memory_size   = 1024

  depends_on = [aws_iam_role_policy.lambda_policy]
}

# Allow S3 to invoke Lambda
resource "aws_lambda_permission" "allow_s3" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ocr_processor.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.documents.arn
}

# Configure S3 notifications to invoke Lambda on object creation
resource "aws_s3_bucket_notification" "bucket_notification" {
  bucket = aws_s3_bucket.documents.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.ocr_processor.arn
    events              = ["s3:ObjectCreated:*"]
    filter_suffix       = ".pdf"
  }

  lambda_function {
    lambda_function_arn = aws_lambda_function.ocr_processor.arn
    events              = ["s3:ObjectCreated:*"]
    filter_suffix       = ".jpg"
  }
}

resource "aws_cloudwatch_log_group" "lambda_log_group" {
  name              = "/aws/lambda/${aws_lambda_function.ocr_processor.function_name}"
  retention_in_days = 30
}

output "s3_bucket" {
  value = aws_s3_bucket.documents.bucket
}

output "ecr_repo" {
  value = aws_ecr_repository.ocr_service.repository_url
}

output "lambda_arn" {
  value = aws_lambda_function.ocr_processor.arn
}

output "lambda_name" {
  value = aws_lambda_function.ocr_processor.function_name
}