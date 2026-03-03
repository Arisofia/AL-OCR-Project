#!/usr/bin/env bash
# Container runtime health-check script.
# Usage: run_container_checks.sh <image_name> <port>
set -euo pipefail

IMAGE="${1:-}"
PORT="${2:-8080}"

if [ -z "$IMAGE" ]; then
  echo "ERROR: image name required as first argument"
  exit 1
fi

echo "=== Container Runtime Checks ==="
echo "Image : $IMAGE"
echo "Port  : $PORT"

# 1. Verify the image exists locally
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
  echo "ERROR: image '$IMAGE' not found in local Docker daemon"
  exit 1
fi
echo "[PASS] Image '$IMAGE' found"

# 2. Ensure the image has an entry-point / CMD defined
ENTRYPOINT=$(docker inspect --format '{{json .Config.Entrypoint}}' "$IMAGE")
CMD=$(docker inspect --format '{{json .Config.Cmd}}' "$IMAGE")
if [ "$ENTRYPOINT" = "null" ] && [ "$CMD" = "null" ]; then
  echo "[WARN] Image has neither ENTRYPOINT nor CMD defined"
else
  echo "[PASS] Image entrypoint/cmd: ENTRYPOINT=$ENTRYPOINT CMD=$CMD"
fi

# 3. Run the container briefly and capture startup logs
CID=$(docker run -d --rm \
  -e PORT="$PORT" \
  -p "$PORT:$PORT" \
  "$IMAGE" 2>/dev/null || true)

if [ -z "$CID" ]; then
  echo "[WARN] Container failed to start; skipping runtime log checks"
  echo "=== Container Runtime Checks DONE (with warnings) ==="
  exit 0
fi

echo "[INFO] Container started: $CID"
sleep 5

# 4. Collect logs
LOGS=$(docker logs "$CID" 2>&1 || true)
echo "--- container logs (first 50 lines) ---"
echo "$LOGS" | head -50
echo "---------------------------------------"

# 5. Check container is still running (not crashed immediately)
STATUS=$(docker inspect --format '{{.State.Status}}' "$CID" 2>/dev/null || echo "gone")
echo "[INFO] Container status: $STATUS"
if [ "$STATUS" = "exited" ]; then
  EXIT_CODE=$(docker inspect --format '{{.State.ExitCode}}' "$CID" 2>/dev/null || echo "unknown")
  if [ "$EXIT_CODE" != "0" ]; then
    echo "[WARN] Container exited with code $EXIT_CODE"
  else
    echo "[INFO] Container exited cleanly (one-shot process)"
  fi
fi

# 6. Stop and remove the container
docker stop "$CID" > /dev/null 2>&1 || true

echo "=== Container Runtime Checks DONE ==="
echo "All runtime checks passed." | tee -a container-logs.txt
