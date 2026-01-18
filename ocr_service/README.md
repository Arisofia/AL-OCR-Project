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
export TF_VAR_account_id="<YOUR_AWS_ACCOUNT_ID>"
export TF_VAR_s3_bucket_name="al-financial-documents-<YOUR_AWS_ACCOUNT_ID>"
terraform init
terraform apply -auto-approve
```

2. Alternatively, run the helper script (idempotent):

```bash
cd ocr_service
./infra_setup.sh
```

3. Build and push the container image, and update Lambda:

```bash
cd ocr_service
./deploy.sh
```

Notes:
- The Dockerfile uses the AWS Lambda base image and includes Tesseract and language packs.
- The Lambda container's entrypoint is `lambda_handler.handler` and supports S3 event triggers: PDF files use Textract async jobs, images use synchronous analyze_document.
- Reconstruction preprocessor: You can enable the optional reconstruction preprocessor (integrates the `ocr_reconstruct` pipeline) by setting the environment variable `ENABLE_RECONSTRUCTION=true`. Control reconstruction iterations with `RECON_ITERATIONS`.
- For production, secure CI credentials and use a locked-down IAM role and stricter Terraform policies (avoid wildcards in ARNs).

---

## Hugging Face Inference (Router) Support 🔁

This project includes a `HuggingFaceVisionProvider` that talks to the **Hugging Face Inference Router** (`https://router.huggingface.co/models/{model}`) instead of the deprecated `api-inference` host.

- **Environment variable**: set `HUGGING_FACE_HUB_TOKEN` (or `hugging_face_hub_token` via `Settings`) in your environment or CI secrets to authenticate requests.
  - Example (local):
    ```bash
    export HUGGING_FACE_HUB_TOKEN="your-token-here"
    ```
- **Why**: The router host is the current supported endpoint and helps reduce deprecation issues and routing problems.
- **Retries**: The provider implements basic retry/backoff on 429 (rate-limiting). For production workloads, prefer using the official `huggingface_hub` SDK's `InferenceApi` for richer retry and throttling support.

**CI checklist** ✅

- Add `HUGGING_FACE_HUB_TOKEN` to your repository secrets in GitHub (Settings → Secrets → Actions).
- Ensure tests that rely on HF are run with a token available or mocked in CI to avoid hitting rate limits.

If you'd like, I can migrate the provider to `huggingface_hub.InferenceApi` in a follow-up PR and add a short CI check that verifies the presence of this secret.
