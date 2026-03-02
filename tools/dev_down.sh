#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT_DIR}/.run"
BACKEND_PID_FILE="${RUN_DIR}/backend.pid"
FRONTEND_PID_FILE="${RUN_DIR}/frontend.pid"
REDIS_MANAGED_FILE="${RUN_DIR}/redis.managed"

stop_from_pid_file() {
  local pid_file="$1"
  local name="$2"

  if [[ ! -f "${pid_file}" ]]; then
    echo "${name}: not running (no pid file)"
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}")"

  if [[ -z "${pid}" ]]; then
    rm -f "${pid_file}"
    echo "${name}: stale pid file removed"
    return 0
  fi

  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    sleep 0.3
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
    echo "${name}: stopped pid ${pid}"
  else
    echo "${name}: process not found (stale pid ${pid})"
  fi

  rm -f "${pid_file}"
}

stop_from_pid_file "${FRONTEND_PID_FILE}" "frontend"
stop_from_pid_file "${BACKEND_PID_FILE}" "backend"

if [[ -f "${REDIS_MANAGED_FILE}" ]]; then
  redis_container="$(cat "${REDIS_MANAGED_FILE}")"
  if [[ -n "${redis_container}" ]] && command -v docker >/dev/null 2>&1; then
    docker stop "${redis_container}" >/dev/null 2>&1 || true
    echo "redis: stopped managed container ${redis_container}"
  fi
  rm -f "${REDIS_MANAGED_FILE}"
fi
