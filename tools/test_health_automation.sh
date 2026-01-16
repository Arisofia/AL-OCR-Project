#!/bin/bash
# Automated FastAPI server startup and health check test
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_ROOT/.venv"
APP_MODULE="ocr_service.main:app"
PORT=8000

# Activate venv
source "$VENV/bin/activate"


# Set PYTHONPATH to project root for import resolution
export PYTHONPATH="$PROJECT_ROOT/ocr_service:$PROJECT_ROOT:$PYTHONPATH"
# Start FastAPI server in background
uvicorn $APP_MODULE --port $PORT --host 127.0.0.1 &
SERVER_PID=$!

# Wait for server to be ready
for i in {1..10}; do
  if curl -s http://127.0.0.1:$PORT/health | grep -q 'status'; then
    echo "FastAPI server is up."
    break
  fi
  sleep 1
done

# Run health check test
pytest ocr_service/tests/test_health_postdeploy.py -v
TEST_RESULT=$?

# Kill FastAPI server
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true

exit $TEST_RESULT
