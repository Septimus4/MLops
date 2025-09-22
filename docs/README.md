# Project Documentation - Home Credit Default Risk

This repo implements an end-to-end, MLflow-tracked MLOps pipeline using a Python `.venv`.

For setup and detailed run instructions, see the root `README.md`.

## How to Run (quick)

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python mlops_pipeline.py
mlflow ui --backend-store-uri file:./mlruns -p 5000  # optional
```

## Step 1 – Understand Data & Objective
- Data identified: `home-credit-default-risk-DATA/*.csv` loaded in `src/data_prep/data_prep.py`.
- Preparation linked to business: features like `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `YEARS_EMPLOYED` connect to repayment capacity risk.
- Features: engineered in `create_basic_features`; categorical encoding via label or one-hot with rare-category handling.
- Justification: ratios capture affordability; tenure features proxy stability; encoded org types reflect employment sector risk.
- Several models tested: Logistic Regression, RandomForest, LightGBM, XGBoost, MLP in `src/training/model_training.py`.
- Cross-validation: `ModelTrainer.cross_validate_model` with `StratifiedKFold` for all models (custom CV for GBMs).
- Class imbalance: weights (`class_weight='balanced'`) and XGB `scale_pos_weight` based on prevalence.
- Metrics aligned to business: ROC AUC, plus precision/recall/F1 used during threshold tuning (FN>FP).
- Feature validation: baseline importance (`get_feature_importance_baseline`) and SHAP/LIME in `src/explainability/model_explainability.py`.

## Step 2 – Track & Compare Experiments
- Tracking: MLflow via `src/training/mlflow_config.py` with parameters, metrics, models (with signature + input example), and dataset samples.
- Documentation of runs: names like "Full Pipeline", per-stage runs, tags `{pipeline, dataset, objective}` in `mlops_pipeline.py`.
- Evidence: see `mlruns/` or start MLflow UI; artifacts include datasets and feature importance plots (when available).

## Step 3 – Build & Improve Model
- Multiple classification models implemented with company-aligned libraries (sklearn, lightgbm, xgboost, MLP).
- Cross-validation across models for robust assessment.
- Imbalanced metrics: AUC + downstream precision/recall used for business threshold tuning.
- Rigorous comparisons: CV mean/std AUC logged; best model selected by CV.
- Imbalance treatment: class weights + XGB `scale_pos_weight`.
- Feature importance accounted: SHAP (tree/linear), LIME (local), baseline RF importance.

## Step 4 – Optimize & Align to Business
- Hyperparameter optimization: Optuna for all models in `src/optimization/hyperparameter_optimization.py`, results tracked in MLflow.
- Business cost function: implemented in `src/training/thresholding.py` with adjustable `fn_cost > fp_cost`.
- Decision threshold: optimized on validation proba in `mlops_pipeline.py`, metrics and selected threshold logged to MLflow.
- Business value link: metadata and logs include cost weights and resulting confusion-matrix-derived metrics.
- MLP activation justification: default `relu` or `tanh` explored by Optuna; `relu` chosen when it improved PR/F1 under imbalance.

## Repo Structure (key files)
- `mlops_pipeline.py`: Orchestrates prep → train/CV → optimize → explain → threshold → deploy.
- `src/data_prep/data_prep.py`: Loading, cleaning, feature engineering, splits, baseline FI.
- `src/training/model_training.py`: Model training, CV, predictions, feature importance.
- `src/optimization/hyperparameter_optimization.py`: Optuna studies (tracked).
- `src/explainability/model_explainability.py`: SHAP/LIME explanations and comparisons.
- `src/training/mlflow_config.py`: MLflow utilities (auto-nesting, datasets, signatures).
- `src/training/thresholding.py`: Business cost-based threshold optimization.
- `model_registry/`: Saved production model and registry metadata.

## Submission Checklist
- .venv created and activated; dependencies installed from `pyproject.toml`
- `home-credit-default-risk-DATA/` present at repo root
- `python mlops_pipeline.py` run completed without errors
- MLflow runs visible in `mlruns/` or via UI; artifacts include model with signature
- `model_registry/best_model_summary.json` present; registry updated with deployment

## Evidence Pointers
- MLflow UI: experiment “Home Credit Default Risk” with nested runs
- Artifacts: dataset samples, feature importance plots, SHAP/LIME outputs (when generated)
- Thresholding: MLflow run “Threshold Optimization” with selected threshold and costs
- Deployment: `model_registry/registry.json` and latest model pickle show promotion
