#!/usr/bin/env bash
set -euo pipefail

# Usage: export AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, (optional AWS_SESSION_TOKEN)
#        export TF_VAR_account_id, TF_VAR_aws_region (default us-east-1)
# Run this script from the repository root: ./infra/apply-oidc.sh

: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID must be set in env}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY must be set in env}"
: "${TF_VAR_account_id:?TF_VAR_account_id must be set in env}"
: "${TF_VAR_aws_region:=us-east-1}"

echo "Using AWS_ACCOUNT: ${TF_VAR_account_id} and region: ${TF_VAR_aws_region}"

cd terraform

echo "Initializing terraform..."
terraform init

echo "Generating plan (saved to plan.tfplan)"
terraform plan -var "account_id=${TF_VAR_account_id}" -var "aws_region=${TF_VAR_aws_region}" -out=plan.tfplan

echo "Showing plan summary..."
terraform show -no-color plan.tfplan | sed -n '1,200p'

read -p "Apply the plan? Type 'yes' to apply: " confirm
if [[ "$confirm" != "yes" ]]; then
  echo "Aborting apply. No changes made."
  exit 0
fi

terraform apply "plan.tfplan"

echo "Fetching outputs..."
terraform output -raw github_actions_role_arn || true
terraform output -json > ../infra/terraform-output.json || true

echo "Done. Please set the following GitHub repository secrets:\n- AWS_ROLE_TO_ASSUME: (value from terraform output github_actions_role_arn)\n- AWS_ACCOUNT_ID: ${TF_VAR_account_id}\n\nYou can get the role ARN with:\n  terraform -chdir=terraform output -raw github_actions_role_arn\n\nAfter setting those secrets, open the GitHub Actions run for the 'Build and Deploy Docker Image' workflow and verify the 'Confirm AWS identity (debug)' step succeeds."