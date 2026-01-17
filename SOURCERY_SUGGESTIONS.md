# Sourcery & Lint Suggestions (Automatable / Low-Risk)

This document lists low-risk suggestions identified by Sourcery/linters and manual inspection. The goal is to batch safe fixes and open a PR for review.

## Priority: High (should address first)

- **Import order / E402**
  - Files: `tools/gemini_cli.py`, `tools/manual_test_services.py` (lines referencing `services.*`), others
  - Action: Move imports to top-level where possible; for local dev fallbacks, wrap sys.path fallback in `try/except`.
  - Rationale: Keeps pre-commit (flake8) happy and avoids CI failures.

- **Test warnings and lints**
  - Files: `ocr_service/tests/test_rate_limit_handler.py`, `ocr_service/tests/test_hugging_face_token_setting.py`
  - Action: Use `_unused` param names and `contextlib.suppress` for cache clearing.
  - Rationale: Reduces noisy failures and simplifies CI diagnostics.

## Priority: Medium (safe refactors)

- **Use lazy logging instead of f-strings**
  - Files: `ocr_service/utils/limiter.py` and other logging sites
  - Action: Replace `logger.error(f"... {e}")` with `logger.error("... %s", e)`.

- **Simplify expressions**
  - Files: `ocr_service/modules/active_learning_orchestrator.py` (`return result.data or []`)
  - Action: Replace ternary-like `result.data if result.data else []` with `result.data or []`.

- **Use walrus or named expressions carefully**
  - Files: `ocr_service/lambda_handler.py` – prefer simple checks to avoid unused variables.

## Priority: Low (stylistic / formatting)

- **Replace non-interpolated f-strings with simple strings**
  - Files: `tools/gemini_cli.py` (`print(f"Initializing GeminiVisionProvider...")`)

- **Line-length fixes (>88 chars)**
  - Files: `ocr_service/modules/advanced_recon.py` -- wrap long default prompt string.

- **Misc**: ensure final newline at EOF in `config.py`, `ai_providers.py`, etc.

---

## Suggested Plan
1. Apply automated edits for the High/Medium items in a single commit.
2. Run pre-commit hooks and tests locally in the CI matrix (3.9-3.11) as a follow-up.
3. Open PR titled: "chore: apply Sourcery & lint cleanups (low-risk)" and request review.

If you'd like, I can apply the High/Medium changes automatically and open the PR now.
