# System Architecture

The project follows a modular MLOps architecture where training, inference, and monitoring components share a consistent artifact contract.

## High-Level Diagram

```
┌──────────────────────────┐
│        Training          │
│  mlops_pipeline.py       │
│  ├─ src/data_prep        │
│  ├─ src/training         │
│  ├─ src/optimization     │
│  └─ src/explainability   │
└────────────┬─────────────┘
             │ promotes
             ▼
┌──────────────────────────┐
│     Model Registry       │
│  model_registry/         │
│  ├─ registry.json        │
│  └─ model pickles        │
└────────────┬─────────────┘
             │ loaded by
             ▼
┌──────────────────────────┐
│    FastAPI Scoring API   │
│  api/app.py              │
│  ├─ model_loader.py      │
│  ├─ inference.py         │
│  ├─ schemas.py           │
│  ├─ logging_utils.py     │
│  └─ gradio_ui.py         │
└────────────┬─────────────┘
             │ logs to
             ▼
┌──────────────────────────┐
│  Runtime Data (data/)    │
│  ├─ logs/YYYY-MM-DD.jsonl│
│  ├─ metrics/             │
│  └─ reference/           │
└────────────┬─────────────┘
             │ read by
             ▼
┌──────────────────────────┐
│ Streamlit Monitoring App │
│  monitor/app.py          │
│  ├─ data_access.py       │
│  ├─ metrics.py           │
│  └─ drift_report.py      │
└──────────────────────────┘
```

## Key Modules

- **Training (`src/`):** Data preparation, feature engineering, model training, hyperparameter optimisation, and explainability. The pipeline registers promoted models and writes metadata into `model_registry/registry.json`.
- **Inference (`api/`):** Loads the latest model via `model_loader.get_model()`, validates payloads with `schemas.InputPayload`, computes predictions in `inference.predict_one`, and logs results with `logging_utils.log_prediction`.
- **UX Layer:** The Gradio interface (`api/gradio_ui.py`) wraps the same inference path for exploratory testing without duplicating logic.
- **Monitoring (`monitor/`):** `data_access` loads recent JSONL logs and reference Parquet snapshots, `metrics` aggregates KPIs, and `drift_report` builds Evidently reports plus a JSON summary consumed by the API’s `/metrics` endpoint.
- **Automation (`scripts/`):** Utilities for smoke tests, log compaction, and reference snapshot generation keep operational tasks repeatable.

## Configuration Flow

Runtime settings are centralised in `api/config.Settings`, which reads environment variables for paths, registry options, and monitoring windows. The same configuration is reused by Streamlit to ensure metrics and reference data resolve consistently.

## Artifact Contracts

- **Feature schema:** `artifacts/feature_list.json`, `feature_defaults.json`, and `categorical_mappings.json` guarantee the API and monitoring code mirror the training schema.
- **Model metadata:** Each registry entry includes the `model_path`, `version`, and optional threshold metadata, exposed via `/health`.
- **Drift summary:** `monitor/drift_report.make_drift_report` saves both an HTML report and `data/metrics/drift/latest_drift_report.json`, which the API surfaces through `/metrics`.
