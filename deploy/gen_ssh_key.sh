#!/usr/bin/env bash
set -euo pipefail

# Generates an SSH key pair for deployment and prints instructions
if [ -z "${1-}" ]; then
  KEY_NAME="gh_deploy"
else
  KEY_NAME="$1"
fi

KEY_DIR="$HOME/.ssh"
mkdir -p "$KEY_DIR"
KEY_PATH="$KEY_DIR/$KEY_NAME"

ssh-keygen -t ed25519 -f "$KEY_PATH" -N "" -C "deploy-key@al-ocr"

echo "Public key ready:"
cat "$KEY_PATH.pub"

echo
cat <<'EOF'
Next steps:
1) Add the public key above to the remote server's ~/.ssh/authorized_keys for the deploy user.
2) Add the PRIVATE KEY (contents of the generated file) to the GitHub repository secrets named: DEPLOY_SSH_KEY
   - Use the GitHub UI or: gh secret set DEPLOY_SSH_KEY --body "$(cat $KEY_PATH)"
3) Add the following secrets to GitHub repository settings:
   - DEPLOY_HOST (IP or host)
   - DEPLOY_USER (SSH user)
   - DEPLOY_PORT (usually 22)
   - REDIS_PASSWORD
   - OCR_API_KEY
   - GHCR_PAT (personal access token with read:packages scope for pulling images from GHCR)
   - AWS_ACCESS_KEY_ID (if using S3 storage)
   - AWS_SECRET_ACCESS_KEY
   - AWS_REGION
4) Trigger the deployment workflow in GitHub Actions (Workflows → Deploy to SSH host).
EOF
