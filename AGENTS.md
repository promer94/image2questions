# Repository Guidelines

## Project Structure & Module Organization

- `src/cli.py`: CLI entrypoint
- `src/agent/`: agent logic, prompts, orchestration.
- `src/models/`: Pydantic models and configuration.
- `src/tools/`: image analysis + output generation (JSON/Word).
- `src/utils/`: shared helpers.
- `tests/`: `pytest` suite (`test_*.py`) and `tests/fixtures/` test data.
- `images/`: sample inputs for local/manual runs.
- `.env.example` / `.env`: local environment configuration (do not commit secrets).

## Build, Test, and Development Commands

This repo uses Python 3.12+ and `uv`.

- Install deps: `uv sync`
- Run the CLI:
  - Interactive chat: `uv run src/cli.py interactive`
- Run tests: `uv run pytest`
- Lint (Ruff): `uv run ruff check .`

## Coding Style & Naming Conventions

- Indentation: 4 spaces; keep lines ~100 chars (Ruff is configured with `line-length = 100`).
- Prefer typed, small functions and Pydantic models in `src/models/`.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Imports are enforced/sorted by Ruff (`select = E,F,I,W`).

## Testing Guidelines

- Framework: `pytest` (configured in `pyproject.toml` with `-v --tb=short`).
- File/function naming: `tests/test_*.py`, `test_*` functions.
- Add tests for new tools/flows alongside the closest existing suite (e.g. `tests/test_tools.py`).

## Commit & Pull Request Guidelines

- Commit history favors short, imperative subjects (e.g. “fix test”, “clean up”); keep commits focused and scoped.
- PRs: describe the change, link issues if applicable, include repro steps (sample image + expected JSON/Word output), and update/add tests when behavior changes.

## Security & Configuration Tips

- Keep API keys in `.env` only; update `.env.example` when adding new required variables.
- If using LangSmith/LangGraph locally, ensure `LANGCHAIN_API_KEY`/project vars are set before running `uv run langgraph dev`.
