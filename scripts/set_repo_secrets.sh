#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/set_repo_secrets.sh owner/repo
# Requires: GH CLI authenticated (gh auth login)
# Reads secrets from environment variables and sets them in the repo.

REPO=${1:?"Usage: $0 <owner/repo>"}

declare -A secrets=(
  [AWS_REGION]="${AWS_REGION:-}"
  [ECR_REPOSITORY]="${ECR_REPOSITORY:-}"
  [AWS_LAMBDA_FUNCTION_NAME]="${AWS_LAMBDA_FUNCTION_NAME:-}"
  [AWS_LAMBDA_ROLE_ARN]="${AWS_LAMBDA_ROLE_ARN:-}"
  [ENABLE_LAMBDA_ROLLBACK]="${ENABLE_LAMBDA_ROLLBACK:-false}"
  [ROLLBACK_COLD_MS]="${ROLLBACK_COLD_MS:-2000}"
  [SLACK_WEBHOOK_URL]="${SLACK_WEBHOOK_URL:-}"
)

echo "Setting secrets for repo: $REPO"

for NAME in "${!secrets[@]}"; do
  VAL=${secrets[$NAME]}
  if [ -z "$VAL" ]; then
    echo "Secret $NAME is empty or not set in environment; skipping (you can set manually via GH UI or export and re-run)."
    continue
  fi
  echo "Setting secret: $NAME"
  gh secret set "$NAME" --repo "$REPO" --body "$VAL"
done

echo "Done. Verify in GitHub repo settings -> Secrets & variables -> Actions."
