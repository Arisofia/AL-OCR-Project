#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Provision a deploy SSH user on a remote Linux host.

Usage:
  ./scripts/provision_deploy_user.sh \
    --host <server> \
    --admin-user <sudo_user> \
    --pubkey-file <path_to_public_key> [options]

Options:
  --host <host>              Target host/IP (required)
  --admin-user <user>        SSH user with sudo rights (required)
  --pubkey-file <path>       Public key file to authorize (required)
  --deploy-user <user>       Deploy user to create/update (default: deploy)
  --port <port>              SSH port (default: 22)
  --no-docker-group          Do not add user to docker group
  --help                     Show this help
EOF
}

HOST=""
ADMIN_USER=""
PUBKEY_FILE=""
DEPLOY_USER="deploy"
PORT="22"
ADD_DOCKER_GROUP=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --admin-user)
      ADMIN_USER="${2:-}"
      shift 2
      ;;
    --pubkey-file)
      PUBKEY_FILE="${2:-}"
      shift 2
      ;;
    --deploy-user)
      DEPLOY_USER="${2:-}"
      shift 2
      ;;
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --no-docker-group)
      ADD_DOCKER_GROUP=0
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

if [[ -z "$HOST" || -z "$ADMIN_USER" || -z "$PUBKEY_FILE" ]]; then
  echo "ERROR: --host, --admin-user, and --pubkey-file are required" >&2
  usage
  exit 1
fi

if [[ ! -f "$PUBKEY_FILE" ]]; then
  echo "ERROR: public key file not found: $PUBKEY_FILE" >&2
  exit 1
fi

PUBKEY_B64="$(base64 < "$PUBKEY_FILE" | tr -d '\n')"

echo "Provisioning deploy user '$DEPLOY_USER' on $HOST:$PORT"
ssh -p "$PORT" "$ADMIN_USER@$HOST" \
  DEPLOY_USER="$DEPLOY_USER" \
  PUBKEY_B64="$PUBKEY_B64" \
  ADD_DOCKER_GROUP="$ADD_DOCKER_GROUP" \
  'bash -s' <<'EOF'
set -euo pipefail

if ! id "$DEPLOY_USER" >/dev/null 2>&1; then
  sudo adduser --disabled-password --gecos "" "$DEPLOY_USER"
fi

if [[ "$ADD_DOCKER_GROUP" == "1" ]]; then
  if getent group docker >/dev/null 2>&1; then
    sudo usermod -aG docker "$DEPLOY_USER"
  fi
fi

sudo install -d -m 700 -o "$DEPLOY_USER" -g "$DEPLOY_USER" "/home/$DEPLOY_USER/.ssh"
AUTH_FILE="/home/$DEPLOY_USER/.ssh/authorized_keys"

PUBKEY="$(printf '%s' "$PUBKEY_B64" | base64 --decode)"
if ! sudo grep -qxF "$PUBKEY" "$AUTH_FILE" 2>/dev/null; then
  printf '%s\n' "$PUBKEY" | sudo tee -a "$AUTH_FILE" >/dev/null
fi

sudo chown "$DEPLOY_USER:$DEPLOY_USER" "$AUTH_FILE"
sudo chmod 600 "$AUTH_FILE"

echo "Deploy user '$DEPLOY_USER' provisioned successfully"
EOF
