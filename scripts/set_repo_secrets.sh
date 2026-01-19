#!/usr/bin/env bash

set -uo pipefail  # Do not use -e so we can handle errors explicitly

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


fail=0
for NAME in "${!secrets[@]}"; do
  VAL=${secrets[$NAME]}
  if [ -z "$VAL" ]; then
    echo "ERROR: Secret $NAME is required but not set in environment." >&2
    fail=1
    continue
  fi
  echo "Setting secret: $NAME"
  if ! gh secret set "$NAME" --repo "$REPO" --body "$VAL"; then
    echo "ERROR: Failed to set secret $NAME" >&2
    fail=1
  fi
done

if [ $fail -ne 0 ]; then
  echo "One or more required secrets were not set. Exiting with error." >&2
  exit 1
fi

echo "Done. Verify in GitHub repo settings -> Secrets & variables -> Actions."
