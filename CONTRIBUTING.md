# Contributing

## Reporting issues

Open a GitHub Issue with:
- What you expected vs. what happened
- Exchange, account type (Futures/Spot), and Python version
- Sanitized config (no API keys)

For security vulnerabilities — use GitHub Security Advisories, not public Issues.

## Pull requests

1. Fork the repo and create a branch from `main`
2. Run tests: `pytest tests/`
3. Run linter: `ruff check . && mypy src/`
4. Open a PR with a clear description of what and why

## Code standards

- Python 3.10+, type-annotated, ruff-formatted
- No credentials in code or comments
- No absolute paths
- Dry-run mode must remain the default for all live-trading components

## What we don't accept

- Strategy parameters, signal logic, or alpha-exposing code
- External service integrations without sandboxed test coverage
- Breaking changes to config schema without a migration note
