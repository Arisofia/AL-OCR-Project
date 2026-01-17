# [1.1.0](https://github.com/Arisofia/AL-OCR-Project/compare/v1.0.1...v1.1.0) (2026-01-17)


### Bug Fixes

* **lambda:** include request_id in payload error warning for better observability and test expectations ([1064003](https://github.com/Arisofia/AL-OCR-Project/commit/10640036cfe8d3a55988ceb8e69852c08987db90))
* log headers in mock post to resolve unused argument warning ([de047ea](https://github.com/Arisofia/AL-OCR-Project/commit/de047ea0327b8ec68b0c69bb944767661220e1a9))
* typing for response body diagnostics; add mock test helper ([bc897a4](https://github.com/Arisofia/AL-OCR-Project/commit/bc897a4b73e3e2c3b4bb9ba4def6e50a1298c88f))


### Features

* **hf:** add HuggingFaceVisionProvider using router.huggingface.co with retry/backoff; add unit tests ([f99672a](https://github.com/Arisofia/AL-OCR-Project/commit/f99672a07ae6996362888e6ad86a232ee1ee6df9))

## [1.0.1](https://github.com/Arisofia/AL-OCR-Project/compare/v1.0.0...v1.0.1) (2026-01-16)


### Bug Fixes

* **tests:** robust health test; ci: dockle install + set image output; mypy install types ([81f8af6](https://github.com/Arisofia/AL-OCR-Project/commit/81f8af65630b694684b68c7a085c274c3c3237d9))

# 1.0.0 (2026-01-16)


### Bug Fixes

* add concise error log and include request_id for pipeline failures ([6f691dd](https://github.com/Arisofia/AL-OCR-Project/commit/6f691dd20570644eeb365a367d9bfcc973dc3501))
* add missing requirements.txt to root for CI ([8cdb207](https://github.com/Arisofia/AL-OCR-Project/commit/8cdb207ea00dc013afeef04e84d867015f1290d1))
* **api:** ensure get_request_id defined before middleware to avoid NameError/linter warnings ([ad64baa](https://github.com/Arisofia/AL-OCR-Project/commit/ad64baae9c052744912b2cb2e38e799ba4751607))
* **api:** ensure get_request_id defined before middleware to avoid NameError/linter warnings ([30dba8d](https://github.com/Arisofia/AL-OCR-Project/commit/30dba8d858075c2a02c6943216ac41e90f93867e))
* **api:** ensure get_request_id defined before middleware to avoid NameError/linter warnings ([bb9aa77](https://github.com/Arisofia/AL-OCR-Project/commit/bb9aa776ff380334c00d068939b7bef6d4f53079))
* **ci:** correct env block and add comments to DVC steps in cml.yaml for clarity and maintainability ([c4a38dc](https://github.com/Arisofia/AL-OCR-Project/commit/c4a38dca6201a63b9c325a8ba1708dc30e9d8b8d))
* **docker:** install shadow-utils and clean yum cache ([f5c8659](https://github.com/Arisofia/AL-OCR-Project/commit/f5c8659c6fc872f23abed52d4fcd0d32f1c84f16))
* ensure critical dependencies are verified in CI ([7cf76f6](https://github.com/Arisofia/AL-OCR-Project/commit/7cf76f69c16fd370d2176ccc9b3ca5c254aca39e))
* **frontend:** validate VITE_API_KEY and only send X-API-KEY when configured ([0bc1faa](https://github.com/Arisofia/AL-OCR-Project/commit/0bc1faacd2f1c6ee916946e60589a3111914f798))
* **frontend:** validate VITE_API_KEY and only send X-API-KEY when configured ([59cfaae](https://github.com/Arisofia/AL-OCR-Project/commit/59cfaaeddd857a2036e08021f2f182bd743ac998))
* handler failure tracking, async learning IO offload, advanced path validation, recon status optional, S3/Textract retry/config and pagination ([9583afd](https://github.com/Arisofia/AL-OCR-Project/commit/9583afd474df3c2b4af95c686fda18f09661d100))
* **lambda:** propagate processing errors so handler can report partial failures; update tests to expect partial_failure ([243d58c](https://github.com/Arisofia/AL-OCR-Project/commit/243d58c202b36a8302bd33019cb3e68de6de5d32))
* provide OCR_API_KEY and increase timeout for backend startup in E2E ([b076564](https://github.com/Arisofia/AL-OCR-Project/commit/b0765641f71bb4269a53792cb91436487315eece))
* remove dead code and hardcoded AWS Account ID, and update DVC with sample data ([abba4bc](https://github.com/Arisofia/AL-OCR-Project/commit/abba4bceb45560cc928fe91a47b34421bd396378))
* remove editable install flag in multi-stage Docker build ([719e1c9](https://github.com/Arisofia/AL-OCR-Project/commit/719e1c97a2c262cba640b94ce91b3a85b130900d))
* remove explicit System/Application dependencies from app.json; use page ID to avoid localization; fix GH workflow outputs and reconstruction error chaining ([96218a1](https://github.com/Arisofia/AL-OCR-Project/commit/96218a15008537daaa4fd33a018277668848ca35))
* resolve OIDC handshake, enhance traceability, and stabilize CI/CD ([6ee9562](https://github.com/Arisofia/AL-OCR-Project/commit/6ee9562dea7d44166345cf4f664e790d658d796e))
* resolve remaining github actions failures ([31ae5bc](https://github.com/Arisofia/AL-OCR-Project/commit/31ae5bc79afef106891d31a5434be242a4aac4cd))
* restore full dependency installation in CI and unify linting standards ([c9a800d](https://github.com/Arisofia/AL-OCR-Project/commit/c9a800dfdfef23c3f2e5eeb32519d70d58ba8530))
* skip AWS authentication on PRs to prevent workflow failure ([8c0bdda](https://github.com/Arisofia/AL-OCR-Project/commit/8c0bdda41f9118ecaead5ea7e35ce826d9267d31))
* stabilize OIDC handshake, resolve CI validation errors, and enhance diagnostics ([243635c](https://github.com/Arisofia/AL-OCR-Project/commit/243635c1cb65b035cd16ca0e7e73ed12957d3046))
* **test:** resolve response.ok AttributeError and missing __init__.py files in ocr_reconstruct ([253086c](https://github.com/Arisofia/AL-OCR-Project/commit/253086cde83ed10c7194de5900655f7cfc6637fb))
* **tests:** restore 'request' param for slowapi compatibility and silence unused-arg warning ([195f750](https://github.com/Arisofia/AL-OCR-Project/commit/195f7505bebd79a1aae88be0361e4a68b6449c4f))


### Features

* achieve enterprise-grade status for OCR service ([377d8a8](https://github.com/Arisofia/AL-OCR-Project/commit/377d8a84fd84a89677b2eb5b59380b93b9ba5cb8))
* add iterative OCR + pixel reconstruction project scaffold with modules, CLI, tests, and ethics note ([935fe6b](https://github.com/Arisofia/AL-OCR-Project/commit/935fe6b18a0d453fae9088f072f3fce3db945d7d))
* **frontend:** add file upload, preview, and OCR request flow ([6fe08ca](https://github.com/Arisofia/AL-OCR-Project/commit/6fe08caf746ce41232e0fb2c9acc12495e8c7293))
* implement automated active learning loop with data quality gates and drift detection ([88934c7](https://github.com/Arisofia/AL-OCR-Project/commit/88934c73e1aab085fa788027fba1569e8d324110))
* Initial commit for AL Financial OCR Project with production-grade infra ([4843b86](https://github.com/Arisofia/AL-OCR-Project/commit/4843b86235a91b511fc94d41ceb91d7e2ee13fd6))
* integrate reconstruction pipeline as optional preprocessor (ENABLE_RECONSTRUCTION) and add tests ([f98ea07](https://github.com/Arisofia/AL-OCR-Project/commit/f98ea077c0df1cfa09ca64891bf603b68ab1dc9c))
* **presign:** add S3 presigned POST endpoint + schemas and tests ([a09daae](https://github.com/Arisofia/AL-OCR-Project/commit/a09daae0c2be19397f4ba7707cde0c194886966f))

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
