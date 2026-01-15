#!/usr/bin/env bash
set -euo pipefail

# cleanup_ecr.sh
# Interactive helper to preview and delete ECR images. Default behavior: delete untagged images only.
# Usage: ./cleanup_ecr.sh [--repo REPO] [--region REGION] [--mode untagged|older-than|all] [--days N]

REPO="al-ocr-service"
REGION="us-east-1"
MODE="untagged"  # untagged | older-than | all
DAYS=30

print_help() {
  cat <<EOF
Usage: $0 [--repo REPO] [--region REGION] [--mode untagged|older-than|all] [--days N]

By default this previews and offers to delete untagged images in REPO.
--mode untagged   Delete images with no tags (safe cleanup)
--mode older-than Delete images older than N days (requires --days)
--mode all        Delete all images in the repository (dangerous)

Examples:
  $0 --mode untagged
  $0 --mode older-than --days 90
  $0 --mode all
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --repo) REPO="$2"; shift 2;;
    --region) REGION="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --days) DAYS="$2"; shift 2;;
    --help) print_help; exit 0;;
    *) echo "Unknown arg: $1"; print_help; exit 1;;
  esac
done

AWS_PAGER="" # ensure aws CLI doesn't use pager

echo "Repository: $REPO"
echo "Region:     $REGION"
echo "Mode:       $MODE"
if [[ "$MODE" == "older-than" ]]; then
  echo "Days:       $DAYS"
fi

printf '\nPreviewing images...\n\n'
if [[ "$MODE" == "untagged" ]]; then
  aws ecr describe-images --repository-name "$REPO" --region "$REGION" --output json \
    | jq -r '.imageDetails[] | select(.imageTags==null) | [.imageDigest, (.imagePushedAt // "" )] | @tsv' || true
elif [[ "$MODE" == "older-than" ]]; then
  python3 - <<PY
import sys, json, datetime
j = json.load(sys.stdin)
cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=int($DAYS))
for d in j.get('imageDetails', []):
    pushed = d.get('imagePushedAt')
    if not pushed: continue
    pushed_dt = datetime.datetime.fromisoformat(pushed.replace('Z','+00:00'))
    if pushed_dt < cutoff:
        tags = ','.join(d.get('imageTags') or ['<untagged>'])
        print(d['imageDigest'], tags, pushed_dt.isoformat())
PY
elif [[ "$MODE" == "all" ]]; then
  aws ecr list-images --repository-name "$REPO" --region "$REGION" --output json \
    | jq -r '.imageIds[] | [.imageDigest, (.imageTag // "<untagged>")] | @tsv' || true
else
  echo "Invalid mode: $MODE"; exit 1
fi

read -p "Proceed to delete matching images? [y/N]: " yn
if [[ ! "$yn" =~ ^[Yy]$ ]]; then
  echo "Aborted. No changes made."; exit 0
fi

if [[ "$MODE" == "untagged" ]]; then
  aws ecr describe-images --repository-name "$REPO" --region "$REGION" --output json \
    | jq -r '.imageDetails[] | select(.imageTags==null) | .imageDigest' \
    | while read -r digest; do
        echo "Deleting $digest"
        AWS_PAGER="" aws ecr batch-delete-image --repository-name "$REPO" --region "$REGION" --image-ids imageDigest="$digest" || true
      done
elif [[ "$MODE" == "older-than" ]]; then
  aws ecr describe-images --repository-name "$REPO" --region "$REGION" --output json \
    | python3 - <<PY | while read -r digest; do
import sys, json, datetime
j = json.load(sys.stdin)
cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=int($DAYS))
for d in j.get('imageDetails', []):
    pushed = d.get('imagePushedAt')
    if not pushed: continue
    pushed_dt = datetime.datetime.fromisoformat(pushed.replace('Z','+00:00'))
    if pushed_dt < cutoff:
        print(d['imageDigest'])
PY
        echo "Deleting $digest"
        AWS_PAGER="" aws ecr batch-delete-image --repository-name "$REPO" --region "$REGION" --image-ids imageDigest="$digest" || true
      done
elif [[ "$MODE" == "all" ]]; then
  aws ecr list-images --repository-name "$REPO" --region "$REGION" --output json \
    | jq -r '.imageIds[] | .imageDigest' \
    | while read -r digest; do
        echo "Deleting $digest"
        AWS_PAGER="" aws ecr batch-delete-image --repository-name "$REPO" --region "$REGION" --image-ids imageDigest="$digest" || true
      done
fi

echo "Done."
