#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT_DIR}/.run"
BACKEND_PID_FILE="${RUN_DIR}/backend.pid"
FRONTEND_PID_FILE="${RUN_DIR}/frontend.pid"
REDIS_MANAGED_FILE="${RUN_DIR}/redis.managed"
BACKEND_LOG="${RUN_DIR}/backend.log"
FRONTEND_LOG="${RUN_DIR}/frontend.log"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_AUTOSTART="${REDIS_AUTOSTART:-true}"
REDIS_CONTAINER_NAME="${REDIS_CONTAINER_NAME:-alocr-dev-redis}"

mkdir -p "${RUN_DIR}"

is_running() {
  local pid_file="$1"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

wait_for_url() {
  local url="$1"
  local retries="${2:-40}"
  local sleep_s="${3:-0.25}"
  local i
  for ((i = 0; i < retries; i++)); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${sleep_s}"
  done
  return 1
}

extract_ocr_api_key() {
  local key="$1"
  local value

  value="${!key:-}"
  if [[ -n "${value}" ]]; then
    printf '%s' "${value}"
    return 0
  fi

  if [[ -f "${ROOT_DIR}/.env" ]]; then
    awk -F= -v key="${key}" '$1 == key {print substr($0, index($0,$2)); exit}' "${ROOT_DIR}/.env"
    return 0
  fi

  return 1
}

is_local_host() {
  local host="$1"
  [[ "${host}" == "127.0.0.1" || "${host}" == "localhost" || "${host}" == "::1" ]]
}

is_port_open() {
  local host="$1"
  local port="$2"
  if command -v nc >/dev/null 2>&1; then
    nc -z "${host}" "${port}" >/dev/null 2>&1
    return $?
  fi
  (echo >/dev/tcp/"${host}"/"${port}") >/dev/null 2>&1
}

ensure_local_redis() {
  if [[ "${REDIS_AUTOSTART}" != "true" ]]; then
    return 0
  fi
  if ! is_local_host "${REDIS_HOST}"; then
    return 0
  fi
  if is_port_open "${REDIS_HOST}" "${REDIS_PORT}"; then
    return 0
  fi
  if ! command -v docker >/dev/null 2>&1; then
    echo "WARNING: Redis is not reachable on ${REDIS_HOST}:${REDIS_PORT} and docker is unavailable."
    return 0
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "WARNING: Redis is not reachable and docker daemon is unavailable."
    return 0
  fi

  if docker ps -a --format '{{.Names}}' | grep -qx "${REDIS_CONTAINER_NAME}"; then
    docker start "${REDIS_CONTAINER_NAME}" >/dev/null
  else
    if [[ -n "${REDIS_PASSWORD_VALUE}" ]]; then
      docker run -d \
        --name "${REDIS_CONTAINER_NAME}" \
        -p "${REDIS_PORT}:6379" \
        redis:alpine \
        redis-server --requirepass "${REDIS_PASSWORD_VALUE}" >/dev/null
    else
      docker run -d \
        --name "${REDIS_CONTAINER_NAME}" \
        -p "${REDIS_PORT}:6379" \
        redis:alpine >/dev/null
    fi
  fi

  local i
  for ((i = 0; i < 20; i++)); do
    if is_port_open "${REDIS_HOST}" "${REDIS_PORT}"; then
      echo "${REDIS_CONTAINER_NAME}" > "${REDIS_MANAGED_FILE}"
      echo "Started managed Redis container (${REDIS_CONTAINER_NAME})"
      return 0
    fi
    sleep 0.25
  done

  echo "WARNING: Redis container started but ${REDIS_HOST}:${REDIS_PORT} is not reachable yet."
}

OCR_KEY="$(extract_ocr_api_key OCR_API_KEY || true)"
if [[ -z "${OCR_KEY}" ]]; then
  echo "ERROR: OCR_API_KEY not found. Set it in environment or ${ROOT_DIR}/.env"
  exit 1
fi

REDIS_PASSWORD_VALUE="$(extract_ocr_api_key REDIS_PASSWORD || true)"

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

ensure_local_redis

if is_running "${BACKEND_PID_FILE}"; then
  echo "Backend already running (pid $(cat "${BACKEND_PID_FILE}"))"
else
  (
    cd "${ROOT_DIR}"
    backend_env=(
      "OCR_API_KEY=${OCR_KEY}"
      "PYTHONPATH=${ROOT_DIR}"
      "REDIS_HOST=${REDIS_HOST}"
      "REDIS_PORT=${REDIS_PORT}"
    )
    if [[ -n "${REDIS_PASSWORD_VALUE}" ]]; then
      backend_env+=("REDIS_PASSWORD=${REDIS_PASSWORD_VALUE}")
    fi

    env "${backend_env[@]}" nohup "${PYTHON_BIN}" -m uvicorn ocr_service.main:app \
        --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" \
        >"${BACKEND_LOG}" 2>&1 &
    echo $! >"${BACKEND_PID_FILE}"
  )
  echo "Started backend (pid $(cat "${BACKEND_PID_FILE}"))"
fi

if ! wait_for_url "http://${BACKEND_HOST}:${BACKEND_PORT}/health"; then
  echo "ERROR: backend failed health check. See ${BACKEND_LOG}"
  exit 1
fi

if is_running "${FRONTEND_PID_FILE}"; then
  echo "Frontend already running (pid $(cat "${FRONTEND_PID_FILE}"))"
else
  (
    cd "${ROOT_DIR}/frontend"
    VITE_API_BASE="http://${BACKEND_HOST}:${BACKEND_PORT}" \
    VITE_API_KEY="${OCR_KEY}" \
      nohup npm run dev -- --host "${FRONTEND_HOST}" --port "${FRONTEND_PORT}" \
        >"${FRONTEND_LOG}" 2>&1 &
    echo $! >"${FRONTEND_PID_FILE}"
  )
  echo "Started frontend (pid $(cat "${FRONTEND_PID_FILE}"))"
fi

if ! wait_for_url "http://${FRONTEND_HOST}:${FRONTEND_PORT}/"; then
  echo "ERROR: frontend failed startup check. See ${FRONTEND_LOG}"
  exit 1
fi

echo
echo "Services are up:"
echo "Backend:  http://${BACKEND_HOST}:${BACKEND_PORT}/health"
echo "Frontend: http://${FRONTEND_HOST}:${FRONTEND_PORT}/"
echo "Redis:    ${REDIS_HOST}:${REDIS_PORT}"
echo "Logs:"
echo "  tail -f ${BACKEND_LOG}"
echo "  tail -f ${FRONTEND_LOG}"
