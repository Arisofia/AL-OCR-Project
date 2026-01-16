# Secure Docker Image Test Plan — OCR Service (Non-root User)

## Overview
This document defines the Test Plan and CI jobs to validate the Secure Docker Image for the OCR Service, with an emphasis on running the service as a non-root user (appuser), enforcing best-practice security settings, and verifying runtime behavior for Lambda/ECS/Fargate/K8s compatibility.

Assumptions (default):
- Targeted runtimes: AWS Lambda (container images) and ECS/Fargate. Kubernetes compatibility is considered.
- Filesystem writes are limited to `/tmp` and stdout/stderr for logs; `/var/task` is read-only in Lambda.
- No GPU required by default.
- CI will run Trivy and Dockle for scanning and will run runtime checks in a disposable container environment.

Acceptance Criteria (high-level):
- Container process runs as non-root (UID != 0).
- Container can write to `/tmp` as the runtime user and ownership is the runtime UID.
- Health endpoint responds as expected and readiness/liveness checks succeed.
- Trivy scan reports no critical vulnerabilities (policy configurable).
- Dockle/CIS checks produce acceptable results and no high-severity fails.
- Container responds to SIGTERM and exits gracefully.

---

## Test Matrix
- Build-time checks: hadolint (Dockerfile), Dockerfile best practices
- Security scans: Trivy (vulns), Dockle (CIS checks)
- Runtime checks: non-root UID, /tmp read/write, health endpoint, graceful shutdown
- Lambda-specific: image size and cold-start smoke (optional)
- Optional GPU tests: validate device access and group membership (if applicable)

---

## Detailed Test Cases

TC-01 — Non-Root Execution
- Goal: Verify container process runs as non-root
- Steps:
  - Build image: `docker build -t al-ocr-service:ci -f ocr_service/Dockerfile .`
  - Run: `docker run --rm --entrypoint '' al-ocr-service:ci sh -c 'id -u'`
  - Expect: output != `0` (fail if equals `0`)

TC-02 — /tmp Write Permission
- Goal: Verify runtime user can write to `/tmp`
- Steps:
  - `docker run --rm al-ocr-service:ci sh -c 'echo ok > /tmp/ci_test.txt && stat -c "%u:%g" /tmp/ci_test.txt'`
  - Expect: command succeeds and prints numeric uid:gid not equal to `0:0` (or acceptable per policy)

TC-03 — Health Endpoint
- Goal: Service responds to `/health`
- Steps:
  - Run container mapping a port (prefer non-privileged >1024) and call `/health` until success or timeout
  - Expect: HTTP 200 with `status` and other expected JSON keys

TC-04 — Graceful Shutdown
- Goal: Application handles SIGTERM gracefully
- Steps:
  - Start container in detached mode
  - Stop container using `docker stop` and ensure container exits within X seconds
  - Expect: container status is stopped and exit code normal

TC-05 — Trivy Vulnerability Scan
- Goal: No critical vulnerabilities (configurable thresholds)
- Steps:
  - Use Trivy to scan the built image
  - Expect: no CVE with severity `CRITICAL` (policy configurable)

TC-06 — Dockle / CIS
- Goal: Dockle scores acceptable with no high severity failures
- Steps:
  - Run Dockle action or CLI against the image
  - Expect: fail only on severe, otherwise documented exceptions

---

## CI Integration
Add a GitHub Actions workflow that:
- Builds the container image
- Runs `tests/container/run_container_checks.sh` against it
- Runs Trivy scan (using `aquasecurity/trivy-action`)
- Runs Dockle (using `goodwithtech/dockle-action`)
- Uploads scan artifacts for auditing

We will add `./.github/workflows/container-security.yml` (see implementation in repository).

---

## Artifacts Provided
- `TEST_PLAN_Secure_Docker_Image.md` — this test plan
- `./tests/container/run_container_checks.sh` — a script with runtime checks
- `./.github/workflows/container-security.yml` — CI job to run scans and tests

---

## Notes
- Policy decisions (e.g., acceptable severity thresholds) should be set in CI via repository secrets or variables.
- If GPU is required, test cases and the Docker runtime policy will be extended to verify device access.

If this plan looks good, I will commit the CI job and script into the repository and enable the workflow.
