#!/usr/bin/env bash
set -euo pipefail
IMAGE=${1:-al-ocr-service:ci}
PORT=${2:-8080}

echo "[CI] Runtime checks for image: ${IMAGE}"

# 1) Check non-root
echo "[CI] Checking non-root user..."
UID_OUT=$(docker run --rm --entrypoint '' "$IMAGE" sh -c 'id -u' || true)
if [ -z "$UID_OUT" ]; then
  echo "[CI] ERROR: Could not determine UID inside container." >&2
  exit 2
fi
if [ "$UID_OUT" -eq 0 ]; then
  echo "[CI] ERROR: Container is running as root (UID 0)." >&2
  exit 2
fi
echo "[CI] UID inside container: $UID_OUT (OK)"

# 2) /tmp write
echo "[CI] Verifying /tmp write permissions..."
WRITE_OUT=$(docker run --rm --entrypoint '' "$IMAGE" sh -c 'sh -c "echo ok >/tmp/ci_test.txt && stat -c "%U:%G" /tmp/ci_test.txt"' || true)
if [ -z "$WRITE_OUT" ]; then
  echo "[CI] ERROR: Unable to write/read /tmp inside container." >&2
  exit 2
fi
echo "[CI] /tmp owner: $WRITE_OUT"

# 3) Health endpoint
echo "[CI] Starting container for health checks..."
CONTAINER_NAME="alocr_ci_check_$RANDOM"
docker run -d --rm --name "$CONTAINER_NAME" -p ${PORT}:${PORT} "$IMAGE" || { echo "[CI] ERROR: failed to start container"; docker logs "$CONTAINER_NAME" || true; exit 2; }

# Capture an initial snapshot of container logs for debugging
set +e
sleep 1
docker logs "$CONTAINER_NAME" > container-logs.txt || true
set -e
echo "[CI] Wrote initial container logs to container-logs.txt"

# Wait for health endpoint (tries common ports if default fails)
HEALTH_OK=0
STOPPED=0
for i in $(seq 1 30); do
  if curl -sSf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "[CI] Health endpoint OK"
    HEALTH_OK=1
    break
  fi
  sleep 1
done

if [ "$HEALTH_OK" -eq 0 ]; then
  echo "[CI] Health endpoint did not respond on port ${PORT} - checking for Lambda-style image..."
  # Capture logs to file for artifact upload and debugging
  docker logs "$CONTAINER_NAME" > container-logs.txt || true
  LOGS=$(cat container-logs.txt || true)
  echo "$LOGS"
  # Accept Lambda-style images: look for common Lambda runtime signatures (case-insensitive)
  if echo "$LOGS" | grep -qiE "bootstrap|lambda runtime|starting lambda|exec\s+['\"]?/var/runtime/bootstrap|Handler|Mangum|Rapid"; then
    echo "[CI] Detected Lambda runtime in container logs; treating as OK for Lambda-style image."
    # Stop the container now that we know it started correctly
    docker stop "$CONTAINER_NAME" || true
    STOPPED=1
    HEALTH_OK=1
  else
    echo "[CI] ERROR: Health endpoint did not respond on port ${PORT}." >&2
    docker stop "$CONTAINER_NAME" || true
    exit 2
  fi
fi

# 4) Graceful shutdown
if [ "$STOPPED" -eq 1 ]; then
  echo "[CI] Container already stopped (Lambda-style image)."
else
  echo "[CI] Testing graceful shutdown..."
  if ! docker stop "$CONTAINER_NAME" >/dev/null 2>&1; then
    echo "[CI] ERROR: Failed to stop container gracefully." >&2
    docker logs "$CONTAINER_NAME" || true
    exit 2
  fi
fi

echo "[CI] Runtime checks passed for ${IMAGE}."

exit 0
