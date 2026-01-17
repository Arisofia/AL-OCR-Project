
# Contributing Guidelines for AL-OCR-Project

## Getting Started
- Fork the repository and clone your fork.
- Create a new branch for each feature or bugfix.
- Ensure your code passes all tests and pre-commit hooks before submitting a PR.

## Code Style & Quality
- Use `ruff` for Python formatting and linting.
- Type annotations are required for all new Python code (enforced via `mypy`).
- Write docstrings for all public functions and classes.
- Use descriptive commit messages (imperative mood, e.g., "Add OCR post-processing module").

## Testing
- All new features must include unit tests (pytest).
- Run `pytest --cov` to check coverage (>80% required for new code).
- For frontend, use Playwright for E2E tests and ensure `npm run lint` passes.

## Pull Requests
- Reference related issues in your PR description.
- Ensure your branch is up to date with `main` before merging.
- PRs require at least one approval and must pass all CI checks.

## Security & Secrets
- Never commit secrets or credentials. Use environment variables and AWS Secrets Manager.
- Follow [OWASP Top Ten](https://owasp.org/www-project-top-ten/) for secure coding.

## Documentation
- Update README and relevant docs for any user-facing changes.
- Add or update architecture diagrams as needed.

## Contact
For questions, open an issue or contact the maintainers.
