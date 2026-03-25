# Contributing to Clawboss

Thanks for your interest in contributing.

## Setup

```bash
git clone https://github.com/arunvenkatadri/Clawboss.git
cd Clawboss
pip install -e ".[dev]"
```

## Running checks

```bash
# Tests
pytest tests/ -v

# Lint
ruff check .
ruff format --check .

# Type check
mypy clawboss/
```

All three must pass before submitting a PR.

## Making changes

1. Fork the repo and create a branch from `main`.
2. Write tests for any new functionality.
3. Make sure all checks pass.
4. Open a PR with a clear description of what changed and why.

## Design principles

- **Zero dependencies.** Everything in `clawboss/` uses only the Python standard library. If you need an external package, it goes in an example or optional integration.
- **Framework agnostic.** Clawboss wraps tool calls — it doesn't own the agent loop. Don't add framework-specific code to the core.
- **Always return, never raise.** `Supervisor.call()` returns a `SupervisedResult`. Exceptions are for programming errors, not tool failures.
- **Thread safe by default.** Any shared state must be protected by a lock.

## Reporting issues

Open an issue on GitHub. Include:
- What you expected to happen
- What actually happened
- Minimal reproduction steps
- Python version and OS
