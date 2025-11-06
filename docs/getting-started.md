# Getting Started

This guide helps you set up a development environment, fetch data, and run the end-to-end pipeline locally.

## Prerequisites

- Python 3.12 or later
- Access to the Home Credit Default Risk dataset (place it under `home-credit-default-risk-DATA/`)
- Recommended: `pip` ≥ 23, `uv` (optional), Docker (for container testing)

## Environment Setup

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

Alternative with `uv`:

```zsh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .[dev]
```

## Pipeline Execution

1. Ensure the dataset folder `home-credit-default-risk-DATA/` sits at the repository root.
2. Run the pipeline:
   ```zsh
   python mlops_pipeline.py
   ```
3. (Optional) Inspect experiments:
   ```zsh
   mlflow ui --backend-store-uri file:./mlruns -p 5000
   ```

The orchestration script handles:

- Data preparation and feature engineering (`src/data_prep/data_prep.py`)
- Model training and cross-validation (`src/training/model_training.py`)
- Optuna optimisation (`src/optimization/hyperparameter_optimization.py`)
- Explainability (SHAP/LIME) and business-aligned threshold selection
- Model promotion into `model_registry/registry.json`

## Quick API Smoke Test

```zsh
uvicorn api.app:app --host 127.0.0.1 --port 8080 &
python -m scripts.smoke_predict --url http://127.0.0.1:8080/predict
curl http://127.0.0.1:8080/metrics
```

Stop the server with `Ctrl+C` or `pkill uvicorn` once finished.

## Troubleshooting

- Missing dataset → ensure CSV files exist under `home-credit-default-risk-DATA/`.
- Model not found → rerun `mlops_pipeline.py` to repopulate `model_registry/`.
- Permission errors writing logs → configure `LOG_DIR`, `METRICS_DIR`, and `REFERENCE_DIR` environment variables or run from the repository root.
