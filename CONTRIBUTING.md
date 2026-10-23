# Contributing to train_with_reversibility

Thank you for your interest in contributing!

## Development Setup

1. Install `uv` (>= 0.4.0) if not already installed.
2. Clone the repository:
   ```bash
   git clone https://github.com/thinkTinkerGreen/train_with_reversibility.git
   cd train_with_reversibility
   ```

3. Sync dependencies:
   ```bash
   uv sync --extra dev
   ```
4. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

## Workflow & Code Standards

- Follow Conventional Commits: `feat: ...`, `fix: ...`, `docs: ...`, `chore: ...`, `test: ...`, `refactor: ...`.
- Run formatting and linters before committing:
  ```bash
  uv run ruff check .
  uv run ruff format --check .
  uv run mypy src/
  ```
- Run tests:
  ```bash
  uv run pytest
  ```
- Ensure any Colab notebooks run smoke checks:
  ```bash
  REVLLM_SMOKE=1 uv run pytest --nbmake notebooks/
  ```

## Pull Request Checklist

- [ ] All tests pass locally and in CI.
- [ ] No hardcoded/invented performance numbers in reports or README.
- [ ] Strict types pass for `revllm.reversible` and `revllm.train`.
- [ ] Documentation updated if models or algorithms changed.
