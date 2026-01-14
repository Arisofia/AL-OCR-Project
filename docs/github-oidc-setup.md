# GitHub Actions -> AWS (OIDC) Setup for AL-OCR-Project

This document explains how to configure AWS IAM and GitHub so GitHub Actions can assume a short-lived role via OIDC (no long-lived AWS keys required).

Overview
- Create an AWS IAM OIDC provider for token.actions.githubusercontent.com (if it doesn't already exist)
- Create an IAM role with a trust policy allowing the GitHub OIDC provider and restrict the `sub` claim to this repository
- Attach a policy to the role that allows ECR push and Lambda update
- Set the role ARN in GitHub as a repository secret `AWS_ROLE_TO_ASSUME` and set `AWS_ACCOUNT_ID` secret if not set already

Steps (AWS CLI)
1. Create OIDC provider (only once per AWS account):

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

2. Create IAM role for GitHub Actions (adjust `repo:Arisofia/AL-OCR-Project:*` if you want tighter scope):

Save the trust policy to `github-oidc-trust.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Federated": "arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"},
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:Arisofia/AL-OCR-Project:*"
        },
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        }
      }
    }
  ]
}
```

Then create the role:

```bash
aws iam create-role --role-name al-ocr-github-actions-role --assume-role-policy-document file://github-oidc-trust.json
```

3. Attach a policy (example inline policy):

Save policy as `ci-ecr-lambda-policy.json` then attach via `aws iam put-role-policy` or create managed policy and attach.

Policy actions should include ECR push permissions and Lambda update permissions (see Terraform in this repo under `terraform/modules/ocr_service` for the exact policy JSON).

4. Put the role ARN into GitHub repository secrets:
- `AWS_ROLE_TO_ASSUME` = `arn:aws:iam::<ACCOUNT_ID>:role/al-ocr-github-actions-role`
- `AWS_ACCOUNT_ID` = `<ACCOUNT_ID>`

Repository Settings -> Secrets & variables -> Actions -> New repository secret

5. Validate in GitHub Actions run
- The workflow `Build and Deploy Docker Image` includes a debug step that runs `aws sts get-caller-identity` which should return the assumed role identity if everything is set correctly.

Notes and best practices
- Prefer restricting the trust policy `sub` claim to specific branches or tags for higher security (e.g. `repo:Arisofia/AL-OCR-Project:ref:refs/heads/main`).
- If you use Terraform, apply the changes in `terraform/` to create the provider and role automatically (see `terraform/modules/ocr_service` in this repo).
- Keep `AWS_ROLE_TO_ASSUME` as a repository secret (role ARN is not especially sensitive but keeping it as a secret hides it in logs and UI).

If you'd like, I can also:
- Add a minimal script to `infra/` that runs the `aws` commands for you (needs an operator with AWS admin credentials), or
- Apply with Terraform if you run `terraform apply` in your AWS environment and give me permission to push the changes now.
