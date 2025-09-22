# Home Credit MLOps Pipeline

End-to-end, MLflow-tracked MLOps pipeline for the Home Credit Default Risk problem. The pipeline covers data prep, model training with CV, hyperparameter optimization, explainability, business-aligned thresholding, and lightweight deployment to a local model registry.

## Quick Start

```zsh
# 1) Create and activate a virtual env
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies (PEP 621 via pyproject)
pip install -U pip
pip install -e .

# 3) Run the pipeline
python mlops_pipeline.py

# 4) (Optional) Open MLflow UI
mlflow ui --backend-store-uri file:./mlruns -p 5000
```

Python 3.12+ is required (see `pyproject.toml`). If you prefer uv:

```zsh
pip install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

## Data Requirements

- Expected path: `home-credit-default-risk-DATA/` in the repo root
- Expected files (subset):
	- `application_train.csv`, `application_test.csv`
	- `bureau.csv`, `bureau_balance.csv`
	- `previous_application.csv`, `installments_payments.csv`
	- `credit_card_balance.csv`, `POS_CASH_balance.csv`
	- `HomeCredit_columns_description.csv`

These are referenced by `src/data_prep/data_prep.py`. If the directory is missing, the pipeline exits with an error.

## What the Pipeline Does

- Data prep and feature engineering with categorical encoding and affordability/stability ratios
- Train multiple models (LogReg, RandomForest, LightGBM, XGBoost, MLP)
- Cross-validation across models (GBMs include early stopping)
- Optuna hyperparameter optimization (tracked in MLflow)
- Explainability via SHAP (tree/linear) and LIME
- Business cost–based threshold optimization (FN cost > FP cost)
- Deployment to a simple on-disk registry (`model_registry/`)

Run orchestration lives in `mlops_pipeline.py`.

## Project Structure (key paths)

- `mlops_pipeline.py`: Orchestrates the full flow and logs to MLflow
- `src/data_prep/data_prep.py`: Loading, cleaning, feature engineering, splits
- `src/training/model_training.py`: Training, CV, prediction utilities
- `src/optimization/hyperparameter_optimization.py`: Optuna studies
- `src/explainability/model_explainability.py`: SHAP/LIME explainability
- `src/training/mlflow_config.py`: MLflow utilities (auto-nesting, signatures, datasets)
- `src/training/thresholding.py`: Cost-based decision threshold optimization
- `model_registry/`: Saved models and `registry.json`
- `mlruns/`: MLflow experiment store (local file backend)
- `docs/`: Short documentation and requirement mapping

## Configuration

Adjust the most common knobs directly in code (kept simple on purpose):

- `mlops_pipeline.py`
	- Sample size: `run_data_preparation(sample_size=0.3)`
	- Run tags and experiment name
	- Threshold costs used during optimization: `fn_cost`, `fp_cost`
- `src/training/model_training.py`
	- CV folds, model defaults, class weight usage, early stopping parameters
- `src/optimization/hyperparameter_optimization.py`
	- Search spaces and number of trials per model

## Outputs

- `mlruns/`: runs, params, metrics, artifacts (plots, samples, models)
- `model_registry/`: production model pickle, `registry.json`, `best_model_summary.json`
- `mlops_pipeline.log`: pipeline logs (if configured by your environment)

The final console summary prints best model, CV AUC, features, and deployment status.

## Known Notes & Warnings

- MLflow deprecation notice for `artifact_path`: non-blocking; future polish may switch to the modern `name=` parameter in logging calls.
- LogisticRegression convergence warnings can appear with imbalance; benign for this use.
- SHAP:
	- Tree SHAP supported and logged
	- Linear SHAP can emit a 1-D shape warning on some folds; when in doubt, consult LIME outputs
	- Deep SHAP requires TensorFlow; not installed by default

## Documentation

- For a concise mapping to the MLOps requirements and short run notes, see `docs/README.md`
- For a requirement-by-requirement breakdown with code links, see `docs/requirements-mapping.md`

## Troubleshooting

- Verify your environment: `python --version` shows 3.12+, and you're inside `.venv`
- Ensure the `home-credit-default-risk-DATA/` directory exists at repo root
- If SHAP import errors occur, reinstall: `pip install shap==0.48.0 numba==0.61.2 llvmlite==0.44.0`

## License

No license specified. If you intend to distribute, add an appropriate license file.