Deployment README

Overview
--------
This repository deploys on push to `main` using GitHub Actions.
The deploy workflow (`.github/workflows/deploy.yml`) performs:

1. Checkout repository code
2. Set up Node.js
3. Install dependencies
4. Run `npm run deploy`

`npm run deploy` delegates to `tools/deploy_app.sh`, which validates required
secrets/env vars and then runs `ocr_service/deploy.sh` to build/push the
container image and update Lambda.

Required GitHub Actions secrets
-------------------------------
Add these in **Settings → Secrets and variables → Actions**:

- `OCR_API_KEY` (mapped to `API_KEY` in deploy job env)
- `GHCR_PAT` (mapped to `ACCESS_TOKEN` in deploy job env)
- `AWS_ACCOUNT_ID`
- `AWS_ROLE_TO_ASSUME`
- `AWS_REGION`

Optional secrets (recommended depending on environment):

- `ECR_REPOSITORY`
- `AWS_LAMBDA_FUNCTION_NAME`
- `AWS_LAMBDA_ROLE_ARN` (used to auto-create Lambda if function is missing)
- `STAGING_API_KEY`
- `NETLIFY_AUTH_TOKEN`

Secret verification
-------------------
You can verify secret presence from local environment variables:

```bash
npm run check:deploy-secrets
# strict mode: fails if required deploy secrets are missing
./tools/check_deploy_secrets.sh --strict
```

CLI automation
--------------
You can generate secrets and upload them to GitHub Actions from CLI:

```bash
# Example: generate secure values, generate SSH key, and set secrets
./scripts/set_repo_secrets.sh \
  --repo owner/repo \
  --generate \
  --generate-ssh-key \
  --set-ssh-secret
```

Notes:
- `GHCR_PAT`, `AWS_ACCOUNT_ID`, `AWS_ROLE_TO_ASSUME`, and `AWS_REGION` must be
  provided as environment variables before running the script.
- Generated values are written to `.deploy-secrets.generated.env` for local
  reuse and are ignored by git.

You can also provision the remote Linux deploy user and authorize the key:

```bash
./scripts/provision_deploy_user.sh \
  --host YOUR_SERVER_IP \
  --admin-user YOUR_SUDO_USER \
  --pubkey-file ./deploy_gha.pub \
  --deploy-user deploy \
  --port 22
```

Notes
-----
- `ocr_service/deploy.sh` now auto-creates the target ECR repository when it
  does not exist.
- If the Lambda function does not exist, deployment auto-creates it when
  `AWS_LAMBDA_ROLE_ARN` is available; otherwise deployment exits with a clear
  error.
- Replace generic env names (`API_KEY`, `ACCESS_TOKEN`) in the workflow only if
  your deployment command expects different variable names.
- Keep secrets in GitHub Actions; never hardcode credentials in the repository.
