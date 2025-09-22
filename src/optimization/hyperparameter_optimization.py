"""
Hyperparameter optimization module using Optuna.
Optimizes hyperparameters for all models with MLflow tracking.
"""

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from typing import Dict, Any, Callable, Optional
import logging

try:
    from training.mlflow_config import MLflowManager
    MLFLOW_AVAILABLE = True
except Exception:
    try:
        from mlflow_config import MLflowManager
        MLFLOW_AVAILABLE = True
    except Exception:
        MLflowManager = None
        MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """Handles hyperparameter optimization using Optuna."""

    def __init__(self, mlflow_manager: Optional[Any] = None,
                 cv_folds: int = 3, random_state: int = 42,
                 n_trials: int = 10, timeout: int = 600):
        """
        Initialize hyperparameter optimizer.

        Args:
            mlflow_manager: MLflow manager for tracking
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            n_trials: Number of optimization trials
            timeout: Timeout for optimization in seconds
        """
        self.mlflow = mlflow_manager
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = {}

    def optimize_logistic_regression(self, X: pd.DataFrame, y: pd.Series,
                                   study_name: str = "logistic_regression_optimization") -> Dict[str, Any]:
        """
        Optimize Logistic Regression hyperparameters.

        Args:
            X: Feature matrix
            y: Target vector
            study_name: Name for the Optuna study

        Returns:
            Best hyperparameters
        """
        logger.info("Optimizing Logistic Regression hyperparameters")

        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'max_iter': 1000,
                'random_state': self.random_state,
                'class_weight': 'balanced'
            }

            # Handle l1 penalty with liblinear solver
            if params['penalty'] == 'l1':
                params['solver'] = 'liblinear'
            else:
                params['solver'] = 'lbfgs'

            model = LogisticRegression(**params)

            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

            return scores.mean()

        study = optuna.create_study(direction='maximize', study_name=study_name)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_params = study.best_params
        best_params.update({
            'max_iter': 1000,
            'random_state': self.random_state,
            'class_weight': 'balanced',
            'solver': 'liblinear' if best_params.get('penalty') == 'l1' else 'lbfgs'
        })

        self.best_params['logistic_regression'] = best_params

        logger.info(f"Best Logistic Regression params: {best_params}")
        logger.info(f"Best CV AUC: {study.best_value:.4f}")

        if self.mlflow:
            with self.mlflow.start_run(run_name="Logistic Regression Optimization"):
                self.mlflow.log_params(best_params)
                self.mlflow.log_metrics({
                    "best_cv_auc": study.best_value,
                    "n_trials": len(study.trials)
                })

        return best_params

    def optimize_random_forest(self, X: pd.DataFrame, y: pd.Series,
                             study_name: str = "random_forest_optimization") -> Dict[str, Any]:
        """
        Optimize Random Forest hyperparameters.

        Args:
            X: Feature matrix
            y: Target vector
            study_name: Name for the Optuna study

        Returns:
            Best hyperparameters
        """
        logger.info("Optimizing Random Forest hyperparameters")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state,
                'class_weight': 'balanced',
                'n_jobs': -1
            }

            model = RandomForestClassifier(**params)

            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

            return scores.mean()

        study = optuna.create_study(direction='maximize', study_name=study_name)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_params = study.best_params
        best_params.update({
            'random_state': self.random_state,
            'class_weight': 'balanced',
            'n_jobs': -1
        })

        self.best_params['random_forest'] = best_params

        logger.info(f"Best Random Forest params: {best_params}")
        logger.info(f"Best CV AUC: {study.best_value:.4f}")

        if self.mlflow:
            with self.mlflow.start_run(run_name="Random Forest Optimization"):
                self.mlflow.log_params(best_params)
                self.mlflow.log_metrics({
                    "best_cv_auc": study.best_value,
                    "n_trials": len(study.trials)
                })

        return best_params

    def optimize_lightgbm(self, X: pd.DataFrame, y: pd.Series,
                        study_name: str = "lightgbm_optimization") -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters.

        Args:
            X: Feature matrix
            y: Target vector
            study_name: Name for the Optuna study

        Returns:
            Best hyperparameters
        """
        logger.info("Optimizing LightGBM hyperparameters")

        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'verbose': -1,
                'random_state': self.random_state
            }

            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

            cv_scores = []
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
                val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=1000,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )

                y_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
                y_pred = np.array(y_pred).flatten()  # Ensure 1D array
                auc = roc_auc_score(y_val_fold, y_pred)
                cv_scores.append(float(auc))

            return float(np.mean(cv_scores))

        study = optuna.create_study(direction='maximize', study_name=study_name)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_params = study.best_params
        best_params.update({
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': self.random_state
        })

        self.best_params['lightgbm'] = best_params

        logger.info(f"Best LightGBM params: {best_params}")
        logger.info(f"Best CV AUC: {study.best_value:.4f}")

        if self.mlflow:
            with self.mlflow.start_run(run_name="LightGBM Optimization"):
                self.mlflow.log_params(best_params)
                self.mlflow.log_metrics({
                    "best_cv_auc": study.best_value,
                    "n_trials": len(study.trials)
                })

        return best_params

    def optimize_xgboost(self, X: pd.DataFrame, y: pd.Series,
                       study_name: str = "xgboost_optimization") -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters.

        Args:
            X: Feature matrix
            y: Target vector
            study_name: Name for the Optuna study

        Returns:
            Best hyperparameters
        """
        logger.info("Optimizing XGBoost hyperparameters")

        # Calculate scale_pos_weight
        pos_weight = len(y[y == 0]) / len(y[y == 1])

        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'random_state': self.random_state,
                'scale_pos_weight': pos_weight
            }

            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

            cv_scores = []
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
                dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=1000,
                    evals=[(dval, 'validation')],
                    early_stopping_rounds=50,
                    verbose_eval=False
                )

                y_pred = model.predict(dval, iteration_range=(0, model.best_iteration))
                y_pred = np.array(y_pred).flatten()  # Ensure 1D array
                auc = roc_auc_score(y_val_fold, y_pred)
                cv_scores.append(float(auc))

            return float(np.mean(cv_scores))

        study = optuna.create_study(direction='maximize', study_name=study_name)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_params = study.best_params
        best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': self.random_state,
            'scale_pos_weight': pos_weight
        })

        self.best_params['xgboost'] = best_params

        logger.info(f"Best XGBoost params: {best_params}")
        logger.info(f"Best CV AUC: {study.best_value:.4f}")

        if self.mlflow:
            with self.mlflow.start_run(run_name="XGBoost Optimization"):
                self.mlflow.log_params(best_params)
                self.mlflow.log_metrics({
                    "best_cv_auc": study.best_value,
                    "n_trials": len(study.trials)
                })

        return best_params

    def optimize_mlp(self, X: pd.DataFrame, y: pd.Series,
                   study_name: str = "mlp_optimization") -> Dict[str, Any]:
        """
        Optimize MLP hyperparameters.

        Args:
            X: Feature matrix
            y: Target vector
            study_name: Name for the Optuna study

        Returns:
            Best hyperparameters
        """
        logger.info("Optimizing MLP hyperparameters")

        def objective(trial):
            n_layers = trial.suggest_int('n_layers', 1, 3)
            hidden_layer_sizes = []

            for i in range(n_layers):
                size = trial.suggest_int(f'hidden_size_{i}', 32, 256)
                hidden_layer_sizes.append(size)

            params = {
                'hidden_layer_sizes': tuple(hidden_layer_sizes),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'solver': 'adam',
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
                'max_iter': 500,
                'random_state': self.random_state,
                'early_stopping': True,
                'validation_fraction': 0.1
            }

            model = MLPClassifier(**params)

            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

            return scores.mean()

        study = optuna.create_study(direction='maximize', study_name=study_name)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_params = study.best_params

        # Reconstruct hidden_layer_sizes
        n_layers = best_params.pop('n_layers')
        hidden_layer_sizes = []
        for i in range(n_layers):
            size = best_params.pop(f'hidden_size_{i}')
            hidden_layer_sizes.append(size)

        best_params['hidden_layer_sizes'] = tuple(hidden_layer_sizes)
        best_params.update({
            'solver': 'adam',
            'max_iter': 500,
            'random_state': self.random_state,
            'early_stopping': True,
            'validation_fraction': 0.1
        })

        self.best_params['mlp'] = best_params

        logger.info(f"Best MLP params: {best_params}")
        logger.info(f"Best CV AUC: {study.best_value:.4f}")

        if self.mlflow:
            with self.mlflow.start_run(run_name="MLP Optimization"):
                self.mlflow.log_params(best_params)
                self.mlflow.log_metrics({
                    "best_cv_auc": study.best_value,
                    "n_trials": len(study.trials)
                })

        return best_params

    def optimize_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Optimize hyperparameters for all models.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Dictionary with best parameters for all models
        """
        logger.info("Optimizing hyperparameters for all models")

        results = {}

        results['logistic_regression'] = self.optimize_logistic_regression(X, y)
        results['random_forest'] = self.optimize_random_forest(X, y)
        results['lightgbm'] = self.optimize_lightgbm(X, y)
        results['xgboost'] = self.optimize_xgboost(X, y)
        results['mlp'] = self.optimize_mlp(X, y)

        return results

    def get_best_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get best parameters for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Best parameters dictionary
        """
        return self.best_params.get(model_name, {})