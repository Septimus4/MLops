# Requirements Mapping (Concise)

## Step 1 – Understand Data & Objective
- Dataset: `home-credit-default-risk-DATA/*.csv` (see `src/data_prep/data_prep.py:DataPreprocessor.load_data`).
- Prep choices ↔ business: affordability ratios, employment/registration tenure (risk of default proxy).
- Features: engineered via `create_basic_features` (same file); categorical encoding with rare-category grouping.
- Justification: documented in code comments and README rationale.
- Models tried: LR, RF, LGBM, XGBoost, MLP (`src/training/model_training.py`).
- Cross-validation: `ModelTrainer.cross_validate_model` with `StratifiedKFold` (custom CV for GBMs with early stopping).
- Imbalance: `class_weight='balanced'`, XGBoost `scale_pos_weight`.
- Metrics: ROC AUC + threshold-tuning precision/recall/F1.
- Validation: baseline RF importance + SHAP/LIME global/local (`src/explainability/model_explainability.py`).

## Step 2 – Track & Compare Experiments
- Tracking: MLflow (`src/training/mlflow_config.py`), with auto-nested runs, model signatures, and dataset logging.
- Clear documentation: run names and `mlflow.set_tags` in `mlops_pipeline.py`.
- Evidence: `mlruns/` directory, or `mlflow ui` to view runs/metrics/artifacts.

## Step 3 – Build & Improve Model
- Models: implemented with company-standard libs (sklearn/lightgbm/xgboost/MLP).
- Cross-validation: applied consistently for model selection.
- Imbalanced metrics: threshold tuning uses precision/recall/F1; PR AUC captured.
- Rigorous comparison: CV AUC mean/std across models; best selected.
- Imbalance treatment: class weights, `scale_pos_weight`.
- Feature importance: SHAP (trees/linear), LIME (local), baseline RF FI.

## Step 4 – Optimize & Align to Business
- Hyperparameter optimization: Optuna (`src/optimization/hyperparameter_optimization.py`).
- Business cost function: `src/training/thresholding.py` (`fn_cost > fp_cost`).
- Threshold adjustment: executed in `mlops_pipeline.py` and logged to MLflow under "Threshold Optimization".
- Documentation of value: MLflow logs include cost weights + metrics; deployment metadata captures threshold.
- MLP activation: explored in Optuna; justification via validation metrics under imbalance.

## Evidence and Artifacts
- MLflow runs: `mlruns/` or UI shows parameters, metrics, and artifacts per stage
- Model signatures: check model artifacts in MLflow; input example logged
- Registry: `model_registry/registry.json`, latest `home_credit_model_v_*.pkl`, and `best_model_summary.json`
