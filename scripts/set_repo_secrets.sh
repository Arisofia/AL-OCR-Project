#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Automate generation and upload of GitHub Actions secrets.

Usage:
  ./scripts/set_repo_secrets.sh --repo owner/repo [options]

Options:
  --repo <owner/repo>          GitHub repository (required)
  --generate                    Auto-generate OCR_API_KEY and REDIS_PASSWORD if missing (default: true)
  --no-generate                 Disable automatic secret generation
  --generate-ssh-key            Generate ed25519 key pair for deploy SSH key
  --ssh-key-path <path>         SSH key path for --generate-ssh-key (default: ./deploy_gha)
  --set-ssh-secret              Set DEPLOY_SSH_KEY from generated/read key file
  --env-file-out <path>         Write effective values to env file (default: .deploy-secrets.generated.env)
  --no-env-file                 Do not write env file
  --dry-run                     Print actions without calling `gh secret set`
  --help                        Show this help

Required env/secrets for this repo deployment:
  GHCR_PAT, AWS_ACCOUNT_ID, AWS_ROLE_TO_ASSUME, AWS_REGION

Optional:
  OCR_API_KEY, REDIS_PASSWORD, ECR_REPOSITORY, AWS_LAMBDA_FUNCTION_NAME,
  S3_BUCKET_NAME,
  DEPLOY_HOST, DEPLOY_USER, DEPLOY_PORT, DEPLOY_SSH_KEY, AWS_ACCESS_KEY_ID,
  AWS_SECRET_ACCESS_KEY, AWS_LAMBDA_ROLE_ARN, ENABLE_LAMBDA_ROLLBACK,
  ROLLBACK_COLD_MS, SLACK_WEBHOOK_URL, STAGING_API_KEY, NETLIFY_AUTH_TOKEN,
  NETLIFY_SITE_ID, VERCEL_TOKEN, VERCEL_ORG_ID, VERCEL_PROJECT_ID
EOF
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $cmd" >&2
    exit 1
  fi
}

mask_preview() {
  local value="$1"
  local len=${#value}
  if [[ $len -le 6 ]]; then
    printf "***"
    return
  fi
  printf "%s***%s" "${value:0:3}" "${value: -3}"
}

set_secret() {
  local repo="$1"
  local name="$2"
  local value="$3"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] would set $name"
    return
  fi
  gh secret set "$name" --repo "$repo" --body "$value"
}

REPO=""
GENERATE=1
GENERATE_SSH_KEY=0
SET_SSH_SECRET=0
SSH_KEY_PATH="./deploy_gha"
ENV_FILE_OUT=".deploy-secrets.generated.env"
WRITE_ENV_FILE=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO="${2:-}"
      shift 2
      ;;
    --generate)
      GENERATE=1
      shift
      ;;
    --no-generate)
      GENERATE=0
      shift
      ;;
    --generate-ssh-key)
      GENERATE_SSH_KEY=1
      shift
      ;;
    --ssh-key-path)
      SSH_KEY_PATH="${2:-}"
      shift 2
      ;;
    --set-ssh-secret)
      SET_SSH_SECRET=1
      shift
      ;;
    --env-file-out)
      ENV_FILE_OUT="${2:-}"
      shift 2
      ;;
    --no-env-file)
      WRITE_ENV_FILE=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$REPO" ]]; then
  echo "ERROR: --repo is required" >&2
  usage
  exit 1
fi

require_cmd gh
require_cmd openssl

if [[ "$DRY_RUN" -eq 0 ]]; then
  if ! gh auth status >/dev/null 2>&1; then
    echo "ERROR: gh is not authenticated. Run: gh auth login" >&2
    exit 1
  fi
fi

# Generate secure values where safe/possible.
if [[ "$GENERATE" -eq 1 ]]; then
  export OCR_API_KEY="${OCR_API_KEY:-$(openssl rand -hex 32)}"
  export REDIS_PASSWORD="${REDIS_PASSWORD:-$(openssl rand -base64 48 | tr -d '\n')}"
  export ENABLE_LAMBDA_ROLLBACK="${ENABLE_LAMBDA_ROLLBACK:-false}"
  export ROLLBACK_COLD_MS="${ROLLBACK_COLD_MS:-2000}"
fi

if [[ "$GENERATE_SSH_KEY" -eq 1 ]]; then
  require_cmd ssh-keygen
  if [[ -f "$SSH_KEY_PATH" || -f "$SSH_KEY_PATH.pub" ]]; then
    if [[ "$SET_SSH_SECRET" -eq 1 && -f "$SSH_KEY_PATH" ]]; then
      echo "SSH key path already exists: $SSH_KEY_PATH(.pub) - reusing existing private key"
    else
      echo "ERROR: SSH key path already exists: $SSH_KEY_PATH(.pub)" >&2
      exit 1
    fi
  else
    ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -N "" -C "gha-deploy@$REPO" >/dev/null
    chmod 600 "$SSH_KEY_PATH"
    chmod 644 "$SSH_KEY_PATH.pub"
    echo "Generated SSH keypair:"
    echo "  private: $SSH_KEY_PATH"
    echo "  public : $SSH_KEY_PATH.pub"
  fi
fi

if [[ "$SET_SSH_SECRET" -eq 1 && -z "${DEPLOY_SSH_KEY:-}" ]]; then
  if [[ -f "$SSH_KEY_PATH" ]]; then
    export DEPLOY_SSH_KEY="$(cat "$SSH_KEY_PATH")"
    echo "Loaded DEPLOY_SSH_KEY from: $SSH_KEY_PATH"
  else
    echo "ERROR: --set-ssh-secret requested but private key not found at $SSH_KEY_PATH" >&2
    exit 1
  fi
fi

declare -a required_names=(
  GHCR_PAT
  AWS_ACCOUNT_ID
  AWS_ROLE_TO_ASSUME
  AWS_REGION
)

declare -a optional_names=(
  OCR_API_KEY
  REDIS_PASSWORD
  ECR_REPOSITORY
  AWS_LAMBDA_FUNCTION_NAME
  S3_BUCKET_NAME
  DEPLOY_HOST
  DEPLOY_USER
  DEPLOY_PORT
  DEPLOY_SSH_KEY
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
  AWS_LAMBDA_ROLE_ARN
  ENABLE_LAMBDA_ROLLBACK
  ROLLBACK_COLD_MS
  SLACK_WEBHOOK_URL
  STAGING_API_KEY
  NETLIFY_AUTH_TOKEN
  NETLIFY_SITE_ID
  VERCEL_TOKEN
  VERCEL_ORG_ID
  VERCEL_PROJECT_ID
)

echo "Applying secrets for repo: $REPO"

missing=0
for name in "${required_names[@]}"; do
  value="${!name:-}"
  if [[ -z "$value" ]]; then
    echo "ERROR: missing required value: $name" >&2
    missing=1
    continue
  fi
  echo "setting required secret: $name ($(mask_preview "$value"))"
  set_secret "$REPO" "$name" "$value"
done

if [[ "$missing" -ne 0 ]]; then
  echo "Aborting: required secrets missing." >&2
  exit 1
fi

for name in "${optional_names[@]}"; do
  value="${!name:-}"
  if [[ -z "$value" ]]; then
    continue
  fi
  echo "setting optional secret: $name ($(mask_preview "$value"))"
  set_secret "$REPO" "$name" "$value"
done

if [[ "$WRITE_ENV_FILE" -eq 1 ]]; then
  umask 077
  cat > "$ENV_FILE_OUT" <<EOF
# Generated by scripts/set_repo_secrets.sh
REPO=${REPO}
GHCR_PAT=${GHCR_PAT:-}
OCR_API_KEY=${OCR_API_KEY:-}
REDIS_PASSWORD=${REDIS_PASSWORD:-}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-}
AWS_ROLE_TO_ASSUME=${AWS_ROLE_TO_ASSUME:-}
AWS_REGION=${AWS_REGION:-}
ECR_REPOSITORY=${ECR_REPOSITORY:-}
AWS_LAMBDA_FUNCTION_NAME=${AWS_LAMBDA_FUNCTION_NAME:-}
S3_BUCKET_NAME=${S3_BUCKET_NAME:-}
DEPLOY_HOST=${DEPLOY_HOST:-}
DEPLOY_USER=${DEPLOY_USER:-}
DEPLOY_PORT=${DEPLOY_PORT:-}
NETLIFY_SITE_ID=${NETLIFY_SITE_ID:-}
VERCEL_TOKEN=${VERCEL_TOKEN:-}
VERCEL_ORG_ID=${VERCEL_ORG_ID:-}
VERCEL_PROJECT_ID=${VERCEL_PROJECT_ID:-}
EOF
  echo "Wrote generated values to: $ENV_FILE_OUT"
  echo "Keep this file secure. It is intended for local use only."
fi

echo "Done. Verify in GitHub repository settings -> Secrets and variables -> Actions."
