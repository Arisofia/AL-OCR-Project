# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.

## Environment (development)

This project reads the following Vite environment variables at runtime. For local development create a copy of `.env.development` from the example and update values as needed.

- `VITE_API_BASE` — Base URL for the OCR API (default: `http://localhost:8000`).
- `VITE_API_KEY` — API key used to authenticate requests to the OCR API. For local dev set `VITE_API_KEY` in `frontend/.env.development` (example value: `REPLACE_WITH_STAGING_API_KEY`). For CI, configure a repo secret named `STAGING_API_KEY` and workflows will pick it up.

Example: copy `frontend/.env.development.example` to `frontend/.env.development` and edit the values.

