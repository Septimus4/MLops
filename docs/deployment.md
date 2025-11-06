# Deployment

This section explains how to containerise the scoring stack and publish runtime images to GitHub Container Registry (GHCR) using the repository’s automated workflows.

## Docker Images

### Scoring API (`Dockerfile`)

1. Builder stage installs dependencies into `/install`.
2. Runtime stage copies application code, artifacts, and registry metadata.
3. Default environment variables (`APP_ENV=prod`, `LOG_DIR=/app/data/logs`, etc.) are set and port `8080` is exposed.

Local build and run:

```zsh
docker build -t scoring-api .
docker run --rm -p 8080:8080 -v $PWD/data:/app/data scoring-api
```

### Monitoring Dashboard (`Dockerfile.monitor`)

Packages the Streamlit dashboard with a smaller dependency footprint. Mount `data/` to surface logs and reference artifacts inside the container.

```zsh
docker build -t scoring-monitor -f Dockerfile.monitor .
docker run --rm -p 7860:7860 -v $PWD/data:/app/data scoring-monitor
```

## CI/CD Workflows

- `.github/workflows/ci.yml` runs on pushes and pull requests:
  - Installs the project with `pip install -e .[dev]`
  - Executes `ruff format --check .` to enforce formatting
  - Runs `ruff check` across `api`, `monitor`, `scripts`, and `tests`
  - Executes `pytest --cov=api --cov=monitor --cov-report=term-missing`
  - Builds both Docker images to ensure they remain valid
- `.github/workflows/publish.yml` runs on pushes to `main` (and is available via “Run workflow”) to publish images to GHCR. Two images are produced each time:
  - `ghcr.io/<owner>/<repo>-api:{latest, <git-sha>}`
  - `ghcr.io/<owner>/<repo>-monitor:{latest, <git-sha>}`

The workflow authenticates with the built-in `GITHUB_TOKEN` (requires `packages: write` permission), so no extra secrets are needed.

## Pulling from GHCR

```zsh
docker login ghcr.io -u <github-username>
docker pull ghcr.io/<owner>/<repo>-api:latest
docker run --rm -p 8080:8080 ghcr.io/<owner>/<repo>-api:latest
```

Replace `<owner>/<repo>` with the actual GitHub namespace (e.g. `acme/mlops`). Images can be pinned to a specific commit SHA by using the `:<git-sha>` tag.

## Environment Variables

| Variable | Purpose | Default |
| -------- | ------- | ------- |
| `MODEL_NAME` | Model key in `registry.json` | `home_credit_model` |
| `MODEL_VERSION` | Optional override for version selection | latest |
| `MODEL_PATH` | Direct path to a model file (bypasses registry) | `None` |
| `LOG_DIR` / `METRICS_DIR` / `REFERENCE_DIR` | Runtime storage | `data/logs`, `data/metrics`, `data/reference` |
| `DEFAULT_THRESHOLD` | Fallback decision threshold | `0.5` |

Configure these values in your deployment environment (e.g. Docker `--env`, Kubernetes secrets) as appropriate.

## Release Checklist

- [ ] Pipeline retrained and latest model promoted to `model_registry/`
- [ ] `python -m scripts.generate_openapi` executed to refresh the OpenAPI spec
- [ ] Automated checks (`ruff format`, `ruff check`, `pytest`) pass locally and on CI
- [ ] Docker images build cleanly without cache
- [ ] Publish workflow succeeds and pushes fresh artifacts to GHCR
- [ ] Streamlit dashboard confirms drift metrics using up-to-date logs
