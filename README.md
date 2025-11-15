# Home Credit MLOps Pipeline & Serving Stack

End-to-end MLOps workflow for the Home Credit Default Risk project, expanded with a production-style scoring API, monitoring dashboard, Docker images, and CI/CD automation.

The repository now contains two major layers:

- **Model Development** (existing): data prep, experimentation, MLflow tracking, Optuna tuning, explainability, and thresholding (`mlops_pipeline.py`, `src/*`).
- **Production Serving** (new): FastAPI scoring service with structured logging, a Gradio manual tester, Streamlit + Evidently monitoring, container builds, automated tests, and GitHub Actions workflows.

---

## Quick Start

### 1. Create a local environment

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
# Optional extras used in monitoring
pip install -r requirements-monitor.txt
```

### 2. Prepare a model artifact

Export your preferred model from MLflow or serialize it manually so that the API can read it. Expected layout (default `artifacts/model/`):

```
artifacts/model/
├── model.pkl             # Required
├── preprocessor.pkl      # Optional (any transform with .transform)
├── metadata.json         # Optional (version, threshold, feature names)
└── feature_names.json    # Optional list of column names
```

Quick helper for generating a fresh XGBoost artifact with validation-tuned thresholding:

```zsh
python scripts/export_xgboost_artifact.py --sample-size 0.30 --overwrite
```

The command trains on the Home Credit dataset (respecting `--sample-size`), computes AUC/PR metrics, and exports `model.pkl`, `feature_names.json`, and a populated `metadata.json` into `artifacts/model/`.

Set `MODEL_REGISTRY_PATH` or `MLFLOW_MODEL_URI` in your environment if the artifact lives elsewhere. For quick smoke tests you can enable `ALLOW_STUB_MODEL=1` (loads a deterministic synthetic model).

### 3. Run the FastAPI service

```zsh
uvicorn api.app:app --host 0.0.0.0 --port 8080 --reload
```

Endpoints:

- `GET /health` → readiness + model metadata
- `POST /predict` → validated scoring request
- `GET /metrics` → recent request volume/latency summary

Structured JSONL logs are written to `data/logs/YYYY-MM-DD.jsonl`.

### 4. Launch the Gradio manual tester (optional)

```zsh
python -m api.gradio_ui
```

### 5. Launch the monitoring dashboard

```zsh
streamlit run monitor/app.py
```

The dashboard visualises KPIs, latency trends, score histograms, and (when Evidently is installed) produces HTML drift reports against a reference snapshot stored in `data/reference/`.

---

## Docker Images

- `Dockerfile` → multi-stage build for the API (`uvicorn api.app:app`).<br>
  ```zsh
  docker build -t scoring-api .
  docker run -p 8080:8080 -e APP_ENV=prod scoring-api
  ```
- `Dockerfile.monitor` → Streamlit + Evidently dashboard.<br>
  ```zsh
  docker build -f Dockerfile.monitor -t scoring-monitor .
  docker run -p 7860:7860 scoring-monitor
  ```

Both images mount `/app/data/*` for logs, metrics, and reference data. Override `LOG_DIR`, `METRICS_DIR`, or `REFERENCE_DIR` if you bind mount host storage.

---

## Tests

Pytest covers contract validation, inference behaviour, logging, and monitoring helpers.

```zsh
pytest --cov=api --cov=monitor
```

Fixtures isolate log directories and enable the stub model automatically, so the suite runs without heavy artifacts.

---

## CI/CD Workflows

Located under `.github/workflows/`:

- `ci.yml`
  - Installs dependencies, runs pytest + coverage, builds/pushes Docker images to GHCR on `main`.
  - Images are tagged as `ghcr.io/<owner>/<repo>/scoring-api` and `scoring-monitor`.
- `deploy_spaces.yml`
  - Prepares bundles for Hugging Face Spaces (Docker for API, Streamlit for monitoring) and uploads them when configured.
  - For security reasons the job ships with empty `HF_TOKEN`, `HF_SPACE_API`, and `HF_SPACE_MONITOR` environment variables. Populate them via repository or environment secrets and update the workflow to inject them (e.g. replace the blank values in the `env` block).

---

## Data Logging & Monitoring

- **Runtime logs**: `data/logs/YYYY-MM-DD.jsonl`
  - Fields include `timestamp`, `request_id`, status, customer hash, latency, payload snapshot, and response.
- **Compaction**: `scripts/compaction.py` converts JSONL logs to Parquet rollups (`data/metrics/daily/`).
- **Reference snapshots**: `scripts/make_reference_snapshot.py` builds/refreshes the reference dataset used by Evidently.
- **Smoke testing**: `scripts/smoke_predict.py` issues a demo `/predict` request with canonical payload.

All scripts honour environment variables exposed via `api.config.Settings`.

---

## Repository Structure (excerpt)

```
api/                  # FastAPI service, logging, schemas, Gradio UI
monitor/              # Streamlit dashboard + Evidently helpers
scripts/              # Operational scripts (reference snapshot, compaction, smoke test)
tests/                # Pytest suite for API + monitoring stack
Dockerfile            # API container (multi-stage, slim)
Dockerfile.monitor    # Monitoring container
.github/workflows/    # CI and Spaces deployment pipelines
requirements.txt      # API/runtime dependencies
requirements-monitor.txt  # Monitoring extras (Streamlit + Evidently)
data/logs|metrics|reference
```

Development assets from the original pipeline remain under `src/`, `docs/`, `mlops_pipeline.py`, and `home-credit-default-risk-DATA/`.

---

## Configuration

`api/config.py` resolves settings from environment variables (and `.env` if present). Key knobs:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_REGISTRY_PATH` | `artifacts/model` | Directory containing `model.pkl` & metadata |
| `MLFLOW_MODEL_URI` | unset | Load directly from MLflow when provided |
| `LOG_DIR` | `data/logs` | Destination for JSONL logs |
| `METRICS_DIR` | `data/metrics` | Aggregated metrics and drift reports |
| `REFERENCE_DIR` | `data/reference` | Reference dataset for drift |
| `ALLOW_STUB_MODEL` | `0` | Development fallback model |

These variables apply to both the FastAPI app and the monitoring dashboard.

---

## Development Pipeline

The original experimentation capabilities remain unchanged:

1. Place raw Home Credit CSVs under `home-credit-default-risk-DATA/`.
2. Run `python mlops_pipeline.py` for the full MLflow-tracked workflow.
3. Export the best model to `artifacts/model/` (or log to MLflow and reference via URI).
4. Serve through the new FastAPI layer and observe monitoring output.

---

## Documentation & Further Reading

- `docs/README.md` – requirement mapping and design notes for the training pipeline.
- `docs/requirements-mapping.md` – detailed rubric alignment.
- `site/` – static artefacts published during Part 1/2.

---

## License

This project is released under the MIT License (see `LICENSE`).