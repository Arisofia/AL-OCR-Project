# Contributing to AL-OCR-Project

Thank you for your interest in contributing! We welcome all contributions that help improve document intelligence for financial services.

## Getting Started

1. Fork the repository.
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/AL-OCR-Project.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Set up the development environment using `docker-compose up`.

## Coding Standards

- **Python**: Follow PEP 8. Use `flake8` and `pylint` for linting.
- **Frontend**: Follow React best practices. Use `eslint` for linting.
- **Infrastructure**: Use `terraform fmt` for all HCL files.

## Testing

Before submitting a PR, ensure all tests pass:
- Backend: `pytest ocr-service/tests`
- Frontend: `npm run test:e2e` (requires Playwright)

## Submitting a Pull Request

1. Push your branch to GitHub.
2. Open a Pull Request against the `main` branch.
3. Provide a clear description of the changes and link any related issues.
4. Ensure CI/CD checks pass.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
