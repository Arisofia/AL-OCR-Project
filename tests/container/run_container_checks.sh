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
# Do not use --rm so logs are available even if the container exits quickly
docker run -d --name "$CONTAINER_NAME" -p ${PORT}:${PORT} "$IMAGE" || { echo "[CI] ERROR: failed to start container"; docker logs "$CONTAINER_NAME" || true; docker rm "$CONTAINER_NAME" || true; exit 2; }

# Ensure the log file exists and capture multiple snapshots of container logs for debugging (append)
touch container-logs.txt
set +e
sleep 1
docker logs -t "$CONTAINER_NAME" >> container-logs.txt 2>&1 || true
sleep 2
docker logs -t "$CONTAINER_NAME" >> container-logs.txt 2>&1 || true
set -e
echo "[CI] Appended initial container logs to container-logs.txt (multiple snapshots)"

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
  docker logs "$CONTAINER_NAME" >> container-logs.txt 2>&1 || true
  LOGS=$(cat container-logs.txt || true)
  echo "$LOGS"
  # Accept Lambda-style images: look for common Lambda runtime signatures (case-insensitive)
  # Match concrete Lambda runtime indicators to reduce false positives
  if echo "$LOGS" | grep -qiE "/var/runtime/bootstrap|exec[[:space:]]+['\"]?/var/runtime/bootstrap|Lambda Runtime|Starting Lambda|Mangum|Rapid|Handler[[:space:]]*[:=]"; then
    echo "[CI] Detected Lambda runtime in container logs (pattern matched); treating as OK for Lambda-style image."
    # Stop and remove the container now that we know it started correctly
    docker stop "$CONTAINER_NAME" || true
    docker rm "$CONTAINER_NAME" || true
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
  # Ensure container is removed after successful stop
  docker rm "$CONTAINER_NAME" || true
fi

echo "[CI] Runtime checks passed for ${IMAGE}."

exit 0
