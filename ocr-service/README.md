# OCR Service Deployment

This document describes how to set up the minimal infra and deploy the OCR Lambda container.

Prerequisites:

- AWS CLI configured with an account that has privileges to create resources.
- Docker and buildx installed locally.
- git repo for commit-tag image tagging.


Quick setup (local):

1. Create infra (Terraform):

```bash
cd infra
export TF_VAR_account_id="510701314494"
export TF_VAR_s3_bucket_name="al-financial-documents-510701314494"
terraform init
terraform apply -auto-approve
```

2. Alternatively, run the helper script (idempotent):

```bash
cd ocr-service
./infra_setup.sh
```

3. Build and push the container image, and update Lambda:

```bash
cd ocr-service
./deploy.sh
```

Notes:
- The Dockerfile uses the AWS Lambda base image and includes Tesseract and language packs.
- The Lambda container's entrypoint is `lambda_handler.handler` and supports S3 event triggers: PDF files use Textract async jobs, images use synchronous analyze_document.
- Reconstruction preprocessor: You can enable the optional reconstruction preprocessor (integrates the `ocr_reconstruct` pipeline) by setting the environment variable `ENABLE_RECONSTRUCTION=true`. Control reconstruction iterations with `RECON_ITERATIONS`.
- For production, secure CI credentials and use a locked-down IAM role and stricter Terraform policies (avoid wildcards in ARNs).
