# Changelog

All notable changes to this project will be documented in this file.

## [1.2.0] - 2026-01-15
### Added
- **Structured JSON Logging**: Implemented `python-json-logger` for machine-readable logs across the service.
- **Sentry Integration**: Added real-time error tracking and performance monitoring for FastAPI and AWS Lambda.
- **Enhanced Health Checks**: Added live connectivity verification for S3, Supabase, and AI provider configuration in `/health`.
- **API Documentation Examples**: Enriched FastAPI docs with Pydantic v2 `json_schema_extra` examples.
- **Traceability**: Unified AWS Request ID extraction and propagation for end-to-end cloud observability.
- **Performance Caching**: Introduced local LRU caching for `LearningEngine` pattern retrieval.

### Changed
- Refactored `lambda_handler` to support structured logging and standardized error responses.
- Updated Pydantic models to use modern v2 standards.

### Security
- **Static Analysis**: Integrated `Bandit` into CI/CD for automated security scanning.
- **Dependency Auditing**: Integrated `pip-audit` to prevent vulnerable packages from reaching production.
- **Configuration Hardening**: Implemented `repr=False` for sensitive settings to prevent exposure in logs.
- Replaced insecure `os.system` calls in tests with safe Python function calls.

## [1.1.0] - 2026-01-15
- **Type Safety**: Achieved 100% Mypy compliance across the codebase.
- **Resilience**: Implemented declarative retries for all AWS service interactions using `tenacity`.
- **Infrastructure**: Established LocalStack parity for local development environments.

## [1.0.0] - 2026-01-15
- Project foundation: FastAPI backend, React frontend, AWS integration, Terraform IaC, and CI/CD workflows.
