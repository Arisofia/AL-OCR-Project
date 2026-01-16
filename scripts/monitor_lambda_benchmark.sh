#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/monitor_lambda_benchmark.sh [workflow_file.yml]
# Default workflow file: .github/workflows/container-security.yml
# Requires: GH CLI authenticated and repo access

WORKFLOW_FILE=${1:-container-security.yml}
REPO=${GITHUB_REPO:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}

echo "Monitoring workflow: $WORKFLOW_FILE in repo $REPO"

# Trigger the workflow
RUN_ID=$(gh workflow run "$WORKFLOW_FILE" --repo "$REPO" --ref main --json id -q '.id')
if [ -z "$RUN_ID" ]; then
  echo "Failed to trigger workflow. Exiting." >&2
  exit 2
fi

echo "Triggered run id: $RUN_ID. Polling status..."

# Poll for completion
while true; do
  STATUS=$(gh run view "$RUN_ID" --repo "$REPO" --json status,conclusion -q '.status')
  CONCLUSION=$(gh run view "$RUN_ID" --repo "$REPO" --json status,conclusion -q '.conclusion')
  echo "Status: $STATUS"
  if [[ "$STATUS" == "completed" ]]; then
    echo "Completed with conclusion: $CONCLUSION"
    break
  fi
  sleep 10
done

# Fetch job logs for 'lambda-benchmark' job to inspect rollback/notifications
JOB_ID=$(gh run view "$RUN_ID" --repo "$REPO" --json jobs -q '.jobs[] | select(.name=="lambda-benchmark") | .id') || true
if [ -n "$JOB_ID" ]; then
  echo "Fetching logs for job id: $JOB_ID (lambda-benchmark)"
  gh run view "$RUN_ID" --repo "$REPO" --job "$JOB_ID" --log > lambda_benchmark_job.log || true
  echo "--- lambda_benchmark_job.log ---"
  tail -n 200 lambda_benchmark_job.log || true
else
  echo "No lambda-benchmark job found in this run. Check jobs list:"
  gh run view "$RUN_ID" --repo "$REPO" --json jobs -q '.jobs[] | {name: .name, id: .id}' || true
fi

# Try to download artifact
ARTIFACTS=$(gh run view "$RUN_ID" --repo "$REPO" --json artifacts -q '.artifacts') || true
if [[ "$ARTIFACTS" != "[]" ]]; then
  echo "Artifacts found; downloading 'lambda-benchmark-results' if present..."
  gh run download "$RUN_ID" --repo "$REPO" --name lambda-benchmark-results --dir ./lambda_artifacts || true
  if [ -f ./lambda_artifacts/lambda_benchmark_results.txt ]; then
    echo "--- Lambda benchmark results ---"
    cat ./lambda_artifacts/lambda_benchmark_results.txt
    if grep -q "Rollback performed" ./lambda_artifacts/lambda_benchmark_results.txt || grep -q "rollback_performed" lambda_benchmark_job.log 2>/dev/null; then
      echo "Rollback was performed during this run."
    else
      echo "No rollback detected in logs or artifacts."
    fi
  else
    echo "Artifact lambda_benchmark_results.txt not found. Listing artifacts:"
    ls -la ./lambda_artifacts || true
  fi
else
  echo "No artifacts uploaded for run $RUN_ID"
fi

exit 0
