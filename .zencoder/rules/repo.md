---
description: Repository Information Overview
alwaysApply: true
---

# AL-OCR-Project Information

## Summary
The **AL-OCR-Project** is a sophisticated fintech-oriented system for iterative OCR and document reconstruction. It leverages a modern stack featuring a FastAPI backend, a React frontend, and AWS services (Textract, S3, SQS). The architecture supports active learning via Label Studio integration and automated infrastructure deployment through Terraform.

## Repository Structure
- **`ocr_service/`**: Core FastAPI backend and worker processes for OCR tasks.
- **`frontend/`**: Modern React application built with Vite and Tailwind CSS.
- **`ocr_reconstruct/`**: Research-focused module for pixel-level image reconstruction.
- **`terraform/`**: Infrastructure as Code for AWS resource provisioning.
- **`data/`**: Data management and DVC configuration.
- **`tools/`**: Utility scripts for environment setup and testing.
- **`infra/`**: Additional scripts for infrastructure management.

### Main Repository Components
- **OCR Backend (`ocr_service`)**: Orchestrates OCR workflows, integrates with AWS Textract, and handles Active Learning logic.
- **React UI (`frontend`)**: Provides an interface for document processing and visualization.
- **Research Engine (`ocr_reconstruct`)**: Specializes in image processing and reconstruction tasks.
- **Cloud Infrastructure (`terraform`)**: Defines the target environment on AWS including OIDC and S3/SQS resources.

## Projects

### OCR Service (Backend)
**Configuration File**: `ocr_service/requirements.txt`, `ocr_service/Dockerfile`

#### Language & Runtime
**Language**: Python
**Version**: 3.11
**Build System**: Manual Scripting / Docker
**Package Manager**: pip

#### Dependencies
**Main Dependencies**:
- `fastapi`, `uvicorn`, `boto3`, `pydantic`, `redis`, `opencv-python-headless`, `scikit-learn`, `openai`, `sentry-sdk`

#### Build & Installation
```bash
cd ocr_service
pip install -r requirements.txt
```

#### Docker
**Dockerfile**: `ocr_service/Dockerfile`
**Image**: Based on `public.ecr.aws/lambda/python:3.11`
**Configuration**: Multi-stage build installing Tesseract and runtime dependencies. Configured for both Lambda execution and containerized worker processes.

#### Testing
**Framework**: Pytest
**Test Location**: `ocr_service/tests/`, `tests/`
**Naming Convention**: `test_*.py`
**Run Command**:
```bash
pytest ocr_service/tests/
```

### Frontend (Web UI)
**Configuration File**: `frontend/package.json`

#### Language & Runtime
**Language**: JavaScript / TypeScript
**Version**: Node.js 20
**Build System**: Vite
**Package Manager**: npm

#### Dependencies
**Main Dependencies**:
- `react`, `react-dom`, `axios`, `framer-motion`, `lucide-react`, `tailwindcss`

#### Build & Installation
```bash
cd frontend
npm install
npm run build
```

#### Docker
**Dockerfile**: `frontend/Dockerfile`
**Image**: `node:20-slim` (Build) / `nginx:stable-alpine` (Runtime)
**Configuration**: Multi-stage build serving static assets via Nginx with SSL support.

#### Testing
**Framework**: Playwright (E2E)
**Test Location**: `frontend/tests/`
**Run Command**:
```bash
npm run test:e2e
```

### OCR Reconstruct (Research Module)
**Configuration File**: `ocr_reconstruct/pyproject.toml`, `ocr_reconstruct/setup.py`

#### Language & Runtime
**Language**: Python
**Version**: 3.9+
**Build System**: setuptools
**Package Manager**: pip

#### Dependencies
**Main Dependencies**:
- `opencv-python-headless`, `pytesseract`, `numpy`, `pillow`, `scikit-image`, `scipy`

#### Build & Installation
```bash
pip install ./ocr_reconstruct
```

#### Testing
**Framework**: Pytest
**Run Command**:
```bash
pytest ocr_reconstruct/tests/
```

## Infrastructure & Orchestration

### Docker Compose
A root-level `docker-compose.yml` orchestrates the entire stack, including:
- `api`: FastAPI backend.
- `worker`: Python background worker.
- `redis`: Task queue and cache.
- `frontend`: React UI.
- `label-studio`: Active learning interface.

### Terraform
Located in `terraform/`, manages AWS resources including:
- S3 Buckets for document storage.
- IAM Roles and OIDC configuration for GitHub Actions.
- SQS Queues for OCR processing.

### Key Operations
- **Dev Environment Setup**: `./tools/setup_test_env.sh` automates venv creation and testing.
- **Local Infrastructure**: Uses LocalStack via `localstack_init/` for AWS simulation.
