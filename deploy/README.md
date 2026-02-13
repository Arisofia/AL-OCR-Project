Deployment README

Overview
--------
This repository contains a GitHub Actions-based deployment pipeline that builds and publishes a container image to GHCR and deploys it to an SSH-accessible host using Docker Compose. The solution uses free components only.

Secrets to add in GitHub repository settings (under Settings → Secrets → Actions):
- DEPLOY_HOST: SSH host (IP or hostname)
- DEPLOY_USER: SSH username
- DEPLOY_PORT: SSH port (default: 22)
- DEPLOY_SSH_KEY: Private deploy key (contents of key from gen_ssh_key.sh)
- GHCR_PAT: Personal Access Token with `read:packages` scope for pulling images from GHCR
- REDIS_PASSWORD: Password to secure Redis
- OCR_API_KEY: API key for the OCR service
- AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION (if you use S3)

You can validate secret presence from your local environment before deployment:

```bash
npm run check:deploy-secrets
# strict mode (fails when required vars are missing)
./tools/check_deploy_secrets.sh --strict
```

Generating keys
----------------
Run:

  ./deploy/gen_ssh_key.sh

Copy the printed public key to the remote server's `~/.ssh/authorized_keys` for the `DEPLOY_USER`.
Add the private key to `DEPLOY_SSH_KEY` secret in the GitHub repository.

How deployment works
--------------------
- On push to `main`, `.github/workflows/publish-image.yml` builds and pushes Docker images to GHCR with two tags: `${{ github.sha }}` and `latest`.
- Trigger the `Deploy to SSH host` workflow manually in Actions or pass `image_tag` to deploy a specific tag.
- The deploy step uses SSH to write a `docker-compose.prod.yml` on the remote host using provided secrets and runs `docker compose pull` and `docker compose up -d`.
- A simple health check hits `/health` and the deploy fails if the app doesn't respond in time (action will report failure).

Notes & rollback
----------------
- This first iteration performs a basic deploy with a health check and will fail the job on unhealthy deploys.
- For more sophisticated canary/blue-green rollouts and automated rollbacks we can extend the `deploy` step with temporary containers and dynamic proxy switching (I can add that next).

Support
-------
If you want me to also provision a free VM and configure it, I can provide Terraform snippets for (e.g., Hetzner Cloud/Runscope or FreeTier) — you'll need to provide credentials or run them yourself.
