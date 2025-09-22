"""
Explainability module for model interpretations using SHAP and LIME.
Provides global and local explanations for model predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union
import logging
import os

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from training.mlflow_config import MLflowManager
    MLFLOW_AVAILABLE = True
except ImportError:
    MLflowManager = None
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelExplainer:
    """Provides model explainability using SHAP and LIME."""

    def __init__(self, mlflow_manager: Optional[Any] = None):
        """
        Initialize model explainer.

        Args:
            mlflow_manager: MLflow manager for tracking
        """
        self.mlflow = mlflow_manager
        self.shap_explainer = None
        self.lime_explainer = None

        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
        if not LIME_AVAILABLE:
            logger.warning("LIME not available. Install with: pip install lime")

    def create_shap_explainer(self, model, X_background: pd.DataFrame,
                            model_type: str = 'tree', max_evals: int = 1000):
        """
        Create SHAP explainer for the model.

        Args:
            model: Trained model
            X_background: Background dataset for SHAP
            model_type: Type of model ('tree', 'linear', 'deep')
            max_evals: Maximum evaluations for permutation explainer
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP not available")
            return None

        logger.info(f"Creating SHAP explainer for {model_type} model")

        try:
            if model_type == 'tree':
                self.shap_explainer = shap.TreeExplainer(model)
            elif model_type == 'linear':
                self.shap_explainer = shap.LinearExplainer(model, X_background)
            elif model_type == 'deep':
                self.shap_explainer = shap.DeepExplainer(model, X_background)
            else:
                # Use permutation explainer as fallback
                self.shap_explainer = shap.PermutationExplainer(model.predict_proba, X_background,
                                                              max_evals=max_evals)

            logger.info("SHAP explainer created successfully")
            return self.shap_explainer

        except Exception as e:
            logger.error(f"Failed to create SHAP explainer: {e}")
            return None

    def create_lime_explainer(self, X_train: pd.DataFrame, feature_names: Optional[List[str]] = None,
                            categorical_features: Optional[List[int]] = None, verbose: bool = False):
        """
        Create LIME explainer for tabular data.

        Args:
            X_train: Training data for LIME
            feature_names: List of feature names
            categorical_features: Indices of categorical features
            verbose: Whether to be verbose
        """
        if not LIME_AVAILABLE:
            logger.error("LIME not available")
            return None

        logger.info("Creating LIME explainer")

        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=feature_names or [f'feature_{i}' for i in range(X_train.shape[1])],
                categorical_features=categorical_features or [],
                verbose=verbose,
                mode='classification'
            )

            logger.info("LIME explainer created successfully")
            return self.lime_explainer

        except Exception as e:
            logger.error(f"Failed to create LIME explainer: {e}")
            return None

    def explain_prediction_shap(self, model, X_instance: pd.DataFrame,
                              model_type: str = 'tree', max_evals: int = 1000) -> Optional[Dict[str, Any]]:
        """
        Explain a single prediction using SHAP.

        Args:
            model: Trained model
            X_instance: Single instance to explain
            model_type: Type of model
            max_evals: Maximum evaluations for permutation explainer

        Returns:
            Dictionary with SHAP explanation results
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP not available")
            return None

        try:
            # Create explainer if not exists
            if self.shap_explainer is None:
                # Use a small background dataset
                background_size = min(100, len(X_instance))
                X_background = X_instance.sample(background_size, random_state=42) if len(X_instance) > background_size else X_instance
                self.create_shap_explainer(model, X_background, model_type, max_evals)

            if self.shap_explainer is None:
                return None

            # Calculate SHAP values
            if hasattr(model, 'predict_proba'):
                shap_values = self.shap_explainer.shap_values(X_instance)
                # For binary classification, shap_values might be a list
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Take positive class
            else:
                shap_values = self.shap_explainer.shap_values(X_instance)

            # Get feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            feature_names = X_instance.columns.tolist()

            # Sort by importance
            sorted_idx = np.argsort(feature_importance)[::-1]
            top_features = [(feature_names[i], feature_importance[i]) for i in sorted_idx[:10]]

            result = {
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'top_features': top_features,
                'expected_value': self.shap_explainer.expected_value
            }

            return result

        except Exception as e:
            logger.error(f"Failed to explain prediction with SHAP: {e}")
            return None

    def explain_prediction_lime(self, model, X_instance: pd.DataFrame,
                              num_features: int = 10) -> Optional[Dict[str, Any]]:
        """
        Explain a single prediction using LIME.

        Args:
            model: Trained model
            X_instance: Single instance to explain
            num_features: Number of features to include in explanation

        Returns:
            Dictionary with LIME explanation results
        """
        if not LIME_AVAILABLE or self.lime_explainer is None:
            logger.error("LIME explainer not available")
            return None

        try:
            # Create prediction function
            def predict_proba(X):
                if hasattr(model, 'predict_proba'):
                    return model.predict_proba(X)
                else:
                    # For models without predict_proba
                    preds = model.predict(X)
                    return np.column_stack([1 - preds, preds])

            # Explain instance
            explanation = self.lime_explainer.explain_instance(
                X_instance.values[0],
                predict_proba,
                num_features=num_features
            )

            # Extract feature contributions
            feature_contributions = explanation.as_list()
            prediction_proba = predict_proba(X_instance.values)[0]

            result = {
                'feature_contributions': feature_contributions,
                'prediction_proba': prediction_proba,
                'predicted_class': np.argmax(prediction_proba)
            }

            return result

        except Exception as e:
            logger.error(f"Failed to explain prediction with LIME: {e}")
            return None

    def global_feature_importance_shap(self, model, X_data: pd.DataFrame,
                                     model_type: str = 'tree', max_evals: int = 10000,
                                     filename: str = "shap_summary_plot.png") -> Optional[pd.DataFrame]:
        """
        Calculate global feature importance using SHAP.

        Args:
            model: Trained model
            X_data: Dataset for global explanation
            model_type: Type of model
            max_evals: Maximum evaluations
            filename: Filename for the plot

        Returns:
            DataFrame with global feature importance
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP not available")
            return None

        try:
            # Sample data for efficiency
            sample_size = min(1000, len(X_data))
            X_sample = X_data.sample(sample_size, random_state=42)

            # Create explainer
            background_size = min(100, sample_size)
            X_background = X_sample.sample(background_size, random_state=42)

            explainer = self.create_shap_explainer(model, X_background, model_type, max_evals)
            if explainer is None:
                return None

            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Calculate mean absolute SHAP values
            feature_importance = np.abs(shap_values).mean(axis=0)
            feature_names = X_sample.columns.tolist()

            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)

            # Create summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close()

            # Log to MLflow
            if self.mlflow:
                with self.mlflow.start_run(run_name="SHAP Global Explanation"):
                    self.mlflow.log_artifact(filename)
                    # Log top 20 features
                    for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
                        self.mlflow.log_metrics({f"feature_{i+1}_importance": row['importance']})

            # Clean up
            if os.path.exists(filename):
                os.remove(filename)

            return importance_df

        except Exception as e:
            logger.error(f"Failed to calculate global SHAP importance: {e}")
            return None

    def plot_shap_waterfall(self, model, X_instance: pd.DataFrame,
                          model_type: str = 'tree', max_evals: int = 1000,
                          filename: str = "shap_waterfall_plot.png") -> bool:
        """
        Create SHAP waterfall plot for a single prediction.

        Args:
            model: Trained model
            X_instance: Single instance to explain
            model_type: Type of model
            max_evals: Maximum evaluations
            filename: Filename for the plot

        Returns:
            True if successful, False otherwise
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP not available")
            return False

        try:
            explanation = self.explain_prediction_shap(model, X_instance, model_type, max_evals)
            if explanation is None:
                return False

            shap_values = explanation['shap_values']
            expected_value = explanation['expected_value']

            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=expected_value,
                    data=X_instance.values[0],
                    feature_names=X_instance.columns.tolist()
                ),
                show=False
            )
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close()

            # Log to MLflow
            if self.mlflow:
                with self.mlflow.start_run(run_name="SHAP Local Explanation"):
                    self.mlflow.log_artifact(filename)

            # Clean up
            if os.path.exists(filename):
                os.remove(filename)

            return True

        except Exception as e:
            logger.error(f"Failed to create SHAP waterfall plot: {e}")
            return False

    def compare_models_shap(self, models: Dict[str, Any], X_data: pd.DataFrame,
                          model_types: Dict[str, str] = None,
                          filename: str = "shap_model_comparison.png") -> Optional[pd.DataFrame]:
        """
        Compare feature importance across multiple models using SHAP.

        Args:
            models: Dictionary of trained models
            X_data: Dataset for comparison
            model_types: Dictionary mapping model names to their types
            filename: Filename for the comparison plot

        Returns:
            DataFrame with feature importance comparison
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP not available")
            return None

        try:
            importance_results = {}

            for model_name, model in models.items():
                model_type = model_types.get(model_name, 'tree') if model_types else 'tree'

                importance_df = self.global_feature_importance_shap(
                    model, X_data, model_type,
                    filename=f"shap_{model_name}_summary.png"
                )

                if importance_df is not None:
                    importance_results[model_name] = importance_df.set_index('feature')['importance']

            if not importance_results:
                return None

            # Combine results
            comparison_df = pd.DataFrame(importance_results).fillna(0)

            # Create comparison plot
            plt.figure(figsize=(12, 8))
            comparison_df.head(15).plot(kind='bar', figsize=(12, 8))
            plt.title('Feature Importance Comparison Across Models')
            plt.xlabel('Features')
            plt.ylabel('SHAP Importance')
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(filename, dpi=150)
            plt.close()

            # Log to MLflow
            if self.mlflow:
                with self.mlflow.start_run(run_name="SHAP Model Comparison"):
                    self.mlflow.log_artifact(filename)

            # Clean up
            if os.path.exists(filename):
                os.remove(filename)

            return comparison_df

        except Exception as e:
            logger.error(f"Failed to compare models with SHAP: {e}")
            return None

    def explain_model_decision(self, model, X_instance: pd.DataFrame,
                             model_name: str = "model", use_shap: bool = True,
                             use_lime: bool = True) -> Dict[str, Any]:
        """
        Provide comprehensive explanation for a model decision.

        Args:
            model: Trained model
            X_instance: Instance to explain
            model_name: Name of the model
            use_shap: Whether to use SHAP
            use_lime: Whether to use LIME

        Returns:
            Dictionary with explanations
        """
        logger.info(f"Explaining {model_name} decision")

        result = {
            'model_name': model_name,
            'instance_shape': X_instance.shape,
            'shap_explanation': None,
            'lime_explanation': None
        }

        # SHAP explanation
        if use_shap and SHAP_AVAILABLE:
            result['shap_explanation'] = self.explain_prediction_shap(model, X_instance)

        # LIME explanation
        if use_lime and LIME_AVAILABLE and self.lime_explainer is not None:
            result['lime_explanation'] = self.explain_prediction_lime(model, X_instance)

        return result