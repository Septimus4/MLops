# Testing & CI

This project treats automated verification as a first-class citizen. Use the following guidance to extend or troubleshoot the suite.

## Test Layout

All tests live in `tests/` and rely on Pytest fixtures to isolate state:

- `test_api_contract.py` – Endpoint happy-path and validation coverage.
- `test_inference.py` – Direct checks on feature preparation and scoring.
- `test_model_load.py` – Ensures the registry loader memoises the model.
- `test_logging.py` – Verifies JSONL output structure and metric aggregation.
- `test_monitor_dashboard.py` – Exercises Streamlit rendering and drift report generation with stubs.
- `test_monitoring.py` – Unit tests for monitoring metric functions.

Run the full suite with:

```zsh
pytest -q --cov=api --cov=monitor
```

## Linting

`ruff` enforces formatting and quality rules:

```zsh
ruff check api monitor scripts tests
```

Integrate this command into your development workflow (e.g., pre-commit hook) to catch issues early.

## Coverage Expectations

`pytest --cov=api --cov=monitor --cov-report=term-missing` covers the most critical runtime components. Target ≥85% coverage for modules you modify.

## Continuous Integration

`.github/workflows/ci.yml` executes on pushes and pull requests to `main`:

1. Set up Python 3.12.
2. Install dependencies via `requirements.txt`.
3. Run `ruff check`.
4. Execute the Pytest suite with coverage.
5. Build Docker images (`Dockerfile` and `Dockerfile.monitor`) to validate container health.

Inspect workflow logs in GitHub Actions if a stage fails. Re-run locally with the same commands to reproduce.

## Extending Tests

- Add new fixtures in `tests/conftest.py` to share setup across modules.
- For new endpoints, add contract tests and update the OpenAPI generator.
- Use stub modules (following the Streamlit/Evidently tests) when third-party dependencies are heavy or unavailable during CI.

## CI Secrets

The publish workflow (`publish.yml`) authenticates to GHCR using the default `GITHUB_TOKEN` with `packages: write` permission, so no extra secrets are required. If you publish to additional registries, store the necessary credentials in the repository’s secrets and reference them in the workflow.
