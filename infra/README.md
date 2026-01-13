# Infra Terraform

This directory contains Terraform templates to create the minimal infra for AL OCR service.

Usage:

1. Initialize Terraform

    terraform init

2. Plan (set variables directly or via TF_VAR_*)

    export TF_VAR_account_id="510701314494"
    export TF_VAR_s3_bucket_name="al-financial-documents-510701314494"
    terraform plan

3. Apply

    terraform apply -auto-approve

Notes:
- The IAM policy here is intentionally broad for the prototype (uses "*"); replace with strict ARNs in production.
- The S3 bucket created is versioned and configured with AES256 encryption. Lifecycle rules will expire objects after 365 days.
