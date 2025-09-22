"""
Model training module for Home Credit Default Risk prediction.
Implements various ML models with cross-validation and MLflow tracking.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from typing import Dict, Any, List, Tuple, Optional
import logging
try:
    from training.mlflow_config import MLflowManager
    MLFLOW_AVAILABLE = True
except Exception:
    MLflowManager = None
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles training of various ML models with cross-validation."""

    def __init__(self, mlflow_manager: Optional[Any] = None,
                 cv_folds: int = 3, random_state: int = 42):
        """
        Initialize model trainer.

        Args:
            mlflow_manager: MLflow manager for tracking
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.mlflow = mlflow_manager
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.models = {}
        self.cv_results = {}

    def get_base_params(self) -> Dict[str, Dict[str, Any]]:
        """Get base parameters for all models."""
        return {
            'logistic_regression': {
                'C': 1.0,
                'penalty': 'l2',
                'max_iter': 2000,  # Increased from 1000
                'solver': 'newton-cg',  # More robust solver for l2 penalty
                'random_state': self.random_state,
                'class_weight': 'balanced'
            },
            'random_forest': {
                'n_estimators': 50,  # Reduced from 100
                'max_depth': 8,  # Reduced from 10
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'random_state': self.random_state,
                'class_weight': 'balanced',
                'n_jobs': -1
            },
            'lightgbm': {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': self.random_state
            },
            'xgboost': {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'scale_pos_weight': 1  # Will be updated based on class balance
            },
            'mlp': {
                'hidden_layer_sizes': (50, 25),  # Reduced from (100, 50)
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'batch_size': 'auto',
                'learning_rate': 'constant',
                'learning_rate_init': 0.001,
                'max_iter': 100,  # Reduced from 200
                'random_state': self.random_state,
                'early_stopping': True,
                'validation_fraction': 0.1
            }
        }

    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
                                start_mlflow_run: bool = True) -> LogisticRegression:
        """
        Train Logistic Regression model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            start_mlflow_run: Whether to start MLflow run (default True)

        Returns:
            Trained LogisticRegression model
        """
        logger.info("Training Logistic Regression model")

        params = self.get_base_params()['logistic_regression']

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            y_pred_proba = np.asarray(model.predict_proba(X_val)[:, 1]).ravel()
            auc = roc_auc_score(y_val, y_pred_proba)
            logger.info(f"Validation AUC (LogReg): {auc:.4f}")

            if self.mlflow and start_mlflow_run:
                with self.mlflow.start_run(run_name="Logistic Regression"):
                    self.mlflow.log_input_dataset(pd.concat([X_train, y_train], axis=1).head(100), name="train_sample", context="training")
                    self.mlflow.log_input_dataset(pd.concat([X_val, y_val], axis=1).head(100), name="val_sample", context="validation")
                    self.mlflow.log_params(params)
                    self.mlflow.log_metrics({"val_auc": auc})
                    self.mlflow.log_model(model, "logistic_regression", "sklearn", input_example=X_val.head(5))

        self.models['logistic_regression'] = model
        return model

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
                          start_mlflow_run: bool = True) -> RandomForestClassifier:
        """
        Train Random Forest model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            start_mlflow_run: Whether to start MLflow run (default True)

        Returns:
            Trained RandomForestClassifier model
        """
        logger.info("Training Random Forest model")

        params = self.get_base_params()['random_forest']

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            y_pred_proba = np.asarray(model.predict_proba(X_val)[:, 1]).ravel()
            auc = roc_auc_score(y_val, y_pred_proba)
            logger.info(f"Validation AUC (RF): {auc:.4f}")

            if self.mlflow and start_mlflow_run:
                with self.mlflow.start_run(run_name="Random Forest"):
                    self.mlflow.log_input_dataset(pd.concat([X_train, y_train], axis=1).head(100), name="train_sample", context="training")
                    self.mlflow.log_input_dataset(pd.concat([X_val, y_val], axis=1).head(100), name="val_sample", context="validation")
                    self.mlflow.log_params(params)
                    self.mlflow.log_metrics({"val_auc": auc})
                    self.mlflow.log_model(model, "random_forest", "sklearn", input_example=X_val.head(5))

        self.models['random_forest'] = model
        return model

    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
                     num_boost_round: int = 200, early_stopping_rounds: int = 50,
                     start_mlflow_run: bool = True) -> lgb.Booster:
        """
        Train LightGBM model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Early stopping rounds
            start_mlflow_run: Whether to start MLflow run (default True)

        Returns:
            Trained LightGBM model
        """
        logger.info("Training LightGBM model")

        params = self.get_base_params()['lightgbm']

        train_data = lgb.Dataset(X_train, label=y_train)

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [train_data, val_data]
            valid_names = ['train', 'valid']
        else:
            valid_sets = [train_data]
            valid_names = ['train']

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100)
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            y_pred_proba = np.asarray(model.predict(X_val, num_iteration=model.best_iteration)).ravel()
            auc = roc_auc_score(y_val, y_pred_proba)
            logger.info(f"Validation AUC (LightGBM): {auc:.4f}")

            if self.mlflow and start_mlflow_run:
                with self.mlflow.start_run(run_name="LightGBM"):
                    self.mlflow.log_input_dataset(pd.concat([X_train, y_train], axis=1).head(100), name="train_sample", context="training")
                    self.mlflow.log_input_dataset(pd.concat([X_val, y_val], axis=1).head(100), name="val_sample", context="validation")
                    self.mlflow.log_params(params)
                    self.mlflow.log_params({
                        'num_boost_round': num_boost_round,
                        'early_stopping_rounds': early_stopping_rounds,
                        'best_iteration': model.best_iteration
                    })
                    self.mlflow.log_metrics({"val_auc": auc})
                    self.mlflow.log_model(model, "lightgbm", "lightgbm", input_example=X_val.head(5))

        self.models['lightgbm'] = model
        return model

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
                    num_boost_round: int = 200, early_stopping_rounds: int = 50,
                    start_mlflow_run: bool = True) -> xgb.Booster:
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Early stopping rounds
            start_mlflow_run: Whether to start MLflow run (default True)

        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost model")

        params = self.get_base_params()['xgboost']

        # Update scale_pos_weight based on class balance
        pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        params['scale_pos_weight'] = pos_weight

        dtrain = xgb.DMatrix(X_train, label=y_train)

        dval = None
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            watchlist = [(dtrain, 'train'), (dval, 'valid')]
        else:
            watchlist = [(dtrain, 'train')]

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )

        # Evaluate on validation set if provided
        if dval is not None and X_val is not None and y_val is not None:
            y_pred_proba = np.asarray(model.predict(dval, iteration_range=(0, model.best_iteration))).ravel()
            auc = roc_auc_score(y_val, y_pred_proba)
            logger.info(f"Validation AUC (XGBoost): {auc:.4f}")

            if self.mlflow and start_mlflow_run:
                with self.mlflow.start_run(run_name="XGBoost"):
                    self.mlflow.log_input_dataset(pd.concat([X_train, y_train], axis=1).head(100), name="train_sample", context="training")
                    self.mlflow.log_input_dataset(pd.concat([X_val, y_val], axis=1).head(100), name="val_sample", context="validation")
                    self.mlflow.log_params(params)
                    self.mlflow.log_params({
                        'num_boost_round': num_boost_round,
                        'early_stopping_rounds': early_stopping_rounds,
                        'best_iteration': model.best_iteration
                    })
                    self.mlflow.log_metrics({"val_auc": auc})
                    self.mlflow.log_model(model, "xgboost", "xgboost", input_example=X_val.head(5))

        self.models['xgboost'] = model
        return model

    def train_mlp(self, X_train: pd.DataFrame, y_train: pd.Series,
                X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
                start_mlflow_run: bool = True) -> MLPClassifier:
        """
        Train MLP Classifier model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            start_mlflow_run: Whether to start MLflow run (default True)

        Returns:
            Trained MLPClassifier model
        """
        logger.info("Training MLP Classifier model")

        params = self.get_base_params()['mlp']

        model = MLPClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            y_pred_proba = np.asarray(model.predict_proba(X_val)[:, 1]).ravel()
            auc = roc_auc_score(y_val, y_pred_proba)
            logger.info(f"Validation AUC (MLP): {auc:.4f}")

            if self.mlflow and start_mlflow_run:
                with self.mlflow.start_run(run_name="MLP Classifier"):
                    self.mlflow.log_input_dataset(pd.concat([X_train, y_train], axis=1).head(100), name="train_sample", context="training")
                    self.mlflow.log_input_dataset(pd.concat([X_val, y_val], axis=1).head(100), name="val_sample", context="validation")
                    self.mlflow.log_params(params)
                    self.mlflow.log_metrics({"val_auc": auc})
                    self.mlflow.log_model(model, "mlp", "sklearn", input_example=X_val.head(5))

        self.models['mlp'] = model
        return model

    def cross_validate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series,
                           model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform cross-validation for a specific model.

        Args:
            model_name: Name of the model to cross-validate
            X: Feature matrix
            y: Target vector
            model_params: Model parameters (optional)

        Returns:
            Dictionary with CV results
        """
        logger.info(f"Performing {self.cv_folds}-fold cross-validation for {model_name}")

        if model_params is None:
            model_params = self.get_base_params().get(model_name, {})

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        model = None
        if model_name == 'logistic_regression':
            model = LogisticRegression(**model_params)
        elif model_name == 'random_forest':
            model = RandomForestClassifier(**model_params)
        elif model_name == 'mlp':
            model = MLPClassifier(**model_params)
        elif model_name == 'lightgbm':
            def lgb_cv(Xi, yi):
                skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                scores = []
                params = self.get_base_params()['lightgbm']
                for tr, va in skf.split(Xi, yi):
                    lgb_tr = lgb.Dataset(Xi.iloc[tr], label=yi.iloc[tr])
                    lgb_va = lgb.Dataset(Xi.iloc[va], label=yi.iloc[va], reference=lgb_tr)
                    booster = lgb.train(params, lgb_tr, num_boost_round=300, valid_sets=[lgb_va], callbacks=[lgb.early_stopping(30, verbose=False)])
                    pred = np.asarray(booster.predict(Xi.iloc[va], num_iteration=booster.best_iteration)).ravel()
                    scores.append(roc_auc_score(yi.iloc[va], pred))
                return np.array(scores)
            cv_scores = lgb_cv(X, y)
            results = {'mean_auc': cv_scores.mean(), 'std_auc': cv_scores.std(), 'cv_scores': cv_scores.tolist()}
            self.cv_results[model_name] = results
            logger.info(f"{model_name} CV AUC: {results['mean_auc']:.4f} (+/- {results['std_auc']:.4f})")
            return results
        elif model_name == 'xgboost':
            def xgb_cv(Xi, yi):
                skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                scores = []
                params = self.get_base_params()['xgboost']
                for tr, va in skf.split(Xi, yi):
                    dtr = xgb.DMatrix(Xi.iloc[tr], label=yi.iloc[tr])
                    dva = xgb.DMatrix(Xi.iloc[va], label=yi.iloc[va])
                    booster = xgb.train(params, dtr, num_boost_round=300, evals=[(dva, 'valid')], early_stopping_rounds=30, verbose_eval=False)
                    pred = np.asarray(booster.predict(dva, iteration_range=(0, booster.best_iteration))).ravel()
                    scores.append(roc_auc_score(yi.iloc[va], pred))
                return np.array(scores)
            cv_scores = xgb_cv(X, y)
            results = {'mean_auc': cv_scores.mean(), 'std_auc': cv_scores.std(), 'cv_scores': cv_scores.tolist()}
            self.cv_results[model_name] = results
            logger.info(f"{model_name} CV AUC: {results['mean_auc']:.4f} (+/- {results['std_auc']:.4f})")
            return results
        else:
            raise ValueError(f"Unsupported model for cross-validation: {model_name}")

        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

        results = {
            'mean_auc': cv_scores.mean(),
            'std_auc': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }

        self.cv_results[model_name] = results

        logger.info(f"{model_name} CV AUC: {results['mean_auc']:.4f} (+/- {results['std_auc']:.4f})")

        return results

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train all models and return results.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            Dictionary with trained models and their results
        """
        logger.info("Training all models")

        results = {}

        # Train individual models without starting MLflow runs
        results['logistic_regression'] = self.train_logistic_regression(X_train, y_train, X_val, y_val, start_mlflow_run=False)
        results['random_forest'] = self.train_random_forest(X_train, y_train, X_val, y_val, start_mlflow_run=False)
        results['lightgbm'] = self.train_lightgbm(X_train, y_train, X_val, y_val, start_mlflow_run=False)
        results['xgboost'] = self.train_xgboost(X_train, y_train, X_val, y_val, start_mlflow_run=False)
        results['mlp'] = self.train_mlp(X_train, y_train, X_val, y_val, start_mlflow_run=False)

        # Log all models to MLflow if available
        if self.mlflow:
            with self.mlflow.start_run(run_name="All Models Training"):
                # Log small input dataset samples for traceability
                if X_train is not None and y_train is not None:
                    self.mlflow.log_input_dataset(pd.concat([X_train, y_train], axis=1).head(200), name="train_sample", context="training")
                if X_val is not None and y_val is not None:
                    self.mlflow.log_input_dataset(pd.concat([X_val, y_val], axis=1).head(200), name="val_sample", context="validation")

                # Log parameters for all models
                all_params = {}
                for model_name in results.keys():
                    params = self.get_base_params().get(model_name, {})
                    for param_name, param_value in params.items():
                        all_params[f"{model_name}_{param_name}"] = param_value

                self.mlflow.log_params(all_params)

                # Log each trained model with input_example for signature inference
                input_example = X_val.head(5) if X_val is not None else (X_train.head(5) if X_train is not None else None)
                for model_name, model in results.items():
                    try:
                        if model_name in ['lightgbm', 'xgboost']:
                            self.mlflow.log_model(model, model_name, model_name, input_example=input_example)
                        else:
                            self.mlflow.log_model(model, model_name, "sklearn", input_example=input_example)
                    except Exception as e:
                        logger.warning(f"Failed to log {model_name} to MLflow: {e}")

                # Log summary params
                self.mlflow.log_params({"num_models_trained": len(results)})
                self.mlflow.log_params({"cv_folds": self.cv_folds})

        return results

    def get_model_predictions(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Get predictions from a trained model.

        Args:
            model_name: Name of the model
            X: Feature matrix

        Returns:
            Predicted probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")

        model = self.models[model_name]

        if model_name in ['lightgbm', 'xgboost']:
            if model_name == 'lightgbm':
                return model.predict(X, num_iteration=model.best_iteration)
            else:
                dmatrix = xgb.DMatrix(X)
                return model.predict(dmatrix, iteration_range=(0, model.best_iteration))
        else:
            return model.predict_proba(X)[:, 1]

    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance for a trained model.

        Args:
            model_name: Name of the model
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")

        model = self.models[model_name]

        if model_name == 'random_forest':
            importance = model.feature_importances_
        elif model_name == 'lightgbm':
            importance = model.feature_importance(importance_type='gain')
        elif model_name == 'xgboost':
            importance = model.get_score(importance_type='gain')
            # Convert to array in correct order
            importance = np.array([importance.get(f, 0) for f in feature_names])
        else:
            logger.warning(f"Feature importance not available for {model_name}")
            return pd.DataFrame()

        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return fi_df