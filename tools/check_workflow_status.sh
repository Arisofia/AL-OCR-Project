#!/bin/bash

# Script to monitor the status of a specific workflow run.
# Usage: ./tools/check_workflow_status.sh [RUN_ID] [REPO]
# Defaults: RUN_ID=21083352140, REPO=Arisofia/AL-OCR-Project

RUN_ID=${1:-21083352140}
REPO=${2:-Arisofia/AL-OCR-Project}

for i in {1..40}; do
  s=$(gh run view "$RUN_ID" --repo "$REPO" --json status -q '.status' 2>/dev/null || echo 'missing')
  echo "Attempt $i: status=$s"
  if [ "$s" = "completed" ]; then
    gh run view "$RUN_ID" --repo "$REPO" --json conclusion -q '.conclusion'
    break
  fi
  sleep 5
done
