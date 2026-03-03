#!/usr/bin/env bash

# Script to check all open PRs and their status checks.
# Usage: ./tools/check_all_prs.sh [REPO] [COMMENT_MSG]
# If COMMENT_MSG is provided, it will be posted as a comment to each open PR.

REPO=${1:-Arisofia/AL-OCR-Project}
COMMENT_MSG=${2:-}

echo "Fetching open PRs for $REPO..."

# Check if gh is authenticated
if ! gh auth status >/dev/null 2>&1; then
  echo "Error: GitHub CLI (gh) is not authenticated."
  exit 1
fi

# Get list of open PR numbers
prs=$(gh pr list --repo "$REPO" --state open --json number --jq '.[].number')

if [ -z "$prs" ]; then
  echo "No open PRs found."
  exit 0
fi

for pr in $prs; do
  echo "----------------------------------------------------------------"
  echo "Checking PR #$pr..."
  gh pr view "$pr" --repo "$REPO" --json title,url,state,author,createdAt
  echo ""
  echo "--- Status Checks ---"
  gh pr checks "$pr" --repo "$REPO"

  if [ -n "$COMMENT_MSG" ]; then
    echo "Commenting on PR #$pr..."
    gh pr comment "$pr" --repo "$REPO" --body "$COMMENT_MSG"
  fi
  echo ""
done
