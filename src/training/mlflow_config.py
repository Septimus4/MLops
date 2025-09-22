"""
MLflow configuration and utilities for experiment tracking.
"""

import mlflow
import mlflow.sklearn as mlflow_sklearn
import mlflow.lightgbm as mlflow_lgb
import mlflow.xgboost as mlflow_xgb
from mlflow.models.signature import infer_signature
from typing import Tuple
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MLflowManager:
    """Manages MLflow experiment tracking and logging."""

    def __init__(self, experiment_name: str = "Home Credit Default Risk",
                 tracking_uri: Optional[str] = None):
        """
        Initialize MLflow manager.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (optional)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file:./mlruns"

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Set experiment
        try:
            mlflow.set_experiment(experiment_name)
            logger.info(f"Set MLflow experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Could not set experiment: {e}")

    def start_run(self, run_name: str, nested: bool = False):
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run
            nested: Whether this is a nested run

        Returns:
            MLflow run context
        """
        # Auto-nest when a parent run is already active
        active = mlflow.active_run()
        return mlflow.start_run(run_name=run_name, nested=(nested or active is not None))

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameters to log
        """
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, Any]):
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact to MLflow.

        Args:
            local_path: Local path to the artifact
            artifact_path: Path within the artifact store
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, model, model_name: str, flavor: str = "sklearn", input_example=None, signature=None):
        """
        Log a model to MLflow.

        Args:
            model: The trained model
            model_name: Name for the model
            flavor: MLflow flavor to use
            input_example: Input example for model signature
        """
        try:
            # Prepare input example: cast integer dtypes to float64 to avoid schema warnings
            if input_example is not None:
                try:
                    import pandas as pd
                    if isinstance(input_example, pd.DataFrame):
                        int_cols = input_example.select_dtypes(include=["int", "int32", "int64"]).columns
                        if len(int_cols) > 0:
                            input_example = input_example.copy()
                            input_example[int_cols] = input_example[int_cols].astype("float64")
                except Exception:
                    pass

            # Infer signature if not provided and input_example available
            if signature is None and input_example is not None:
                try:
                    if hasattr(model, 'predict_proba'):
                        preds = model.predict_proba(input_example)
                    else:
                        preds = model.predict(input_example)
                    signature = infer_signature(input_example, preds)
                except Exception:
                    signature = None

            if flavor == "sklearn":
                kwargs = {"sk_model": model, "artifact_path": model_name}
                if input_example is not None:
                    kwargs["input_example"] = input_example
                if signature is not None:
                    kwargs["signature"] = signature
                mlflow_sklearn.log_model(**kwargs)
            elif flavor == "lightgbm":
                kwargs = {"lgb_model": model, "artifact_path": model_name}
                if input_example is not None:
                    kwargs["input_example"] = input_example
                if signature is not None:
                    kwargs["signature"] = signature
                mlflow_lgb.log_model(**kwargs)
            elif flavor == "xgboost":
                kwargs = {"xgb_model": model, "artifact_path": model_name}
                if input_example is not None:
                    kwargs["input_example"] = input_example
                if signature is not None:
                    kwargs["signature"] = signature
                mlflow_xgb.log_model(**kwargs)
            else:
                logger.warning(f"Unsupported flavor: {flavor}")
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")
            # Fallback to basic logging
            mlflow.log_param("model_flavor", flavor)
            mlflow.log_param("model_name", model_name)

    def log_input_dataset(self, df, name: str = "training_data", context: str = "training"):
        """
        Log a pandas DataFrame as an MLflow input dataset.

        Args:
            df: Pandas DataFrame to log
            name: Name of the dataset
            context: Context string (e.g., "training", "validation")
        """
        try:
            # Prefer structured logging when available
            if hasattr(mlflow, "log_table"):
                mlflow.log_table(df, artifact_file=f"datasets/{name}.json")
            else:
                # Fallback: log as CSV artifact
                csv_name = f"{name}.csv"
                df.to_csv(csv_name, index=False)
                mlflow.log_artifact(csv_name, artifact_path="datasets")
                if os.path.exists(csv_name):
                    os.remove(csv_name)
        except Exception as e:
            logger.warning(f"Failed to log dataset: {e}")

    def log_feature_importance(self, feature_names: list, importance_scores: list,
                              filename: str = "feature_importance.csv"):
        """
        Log feature importance as an artifact.

        Args:
            feature_names: List of feature names
            importance_scores: List of importance scores
            filename: Name of the CSV file
        """
        import pandas as pd

        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)

        fi_df.to_csv(filename, index=False)
        mlflow.log_artifact(filename)

        # Clean up
        if os.path.exists(filename):
            os.remove(filename)

    def log_classification_report(self, y_true, y_pred, y_pred_proba=None,
                                filename: str = "classification_report.txt"):
        """
        Log classification report as an artifact.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            filename: Name of the text file
        """
        from sklearn.metrics import classification_report, roc_auc_score

        report = str(classification_report(y_true, y_pred))

        if y_pred_proba is not None:
            auc = roc_auc_score(y_true, y_pred_proba)
            report += f"\nAUC Score: {auc:.4f}"

        with open(filename, 'w') as f:
            f.write(report)

        mlflow.log_artifact(filename)

        # Clean up
        if os.path.exists(filename):
            os.remove(filename)

    def log_confusion_matrix_plot(self, y_true, y_pred,
                                filename: str = "confusion_matrix.png"):
        """
        Log confusion matrix plot as an artifact.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            filename: Name of the plot file
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(filename)
        plt.close()

        mlflow.log_artifact(filename)

        # Clean up
        if os.path.exists(filename):
            os.remove(filename)

    def get_run_info(self):
        """
        Get information about the current run.

        Returns:
            Dictionary with run information
        """
        run = mlflow.active_run()
        if run:
            return {
                'run_id': run.info.run_id,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
                'start_time': run.info.start_time
            }
        return None

    def search_runs(self, filter_string: str = "", order_by: Optional[list] = None):
        """
        Search for runs in the current experiment.

        Args:
            filter_string: Filter string for runs
            order_by: List of columns to order by

        Returns:
            DataFrame with run information
        """
        return mlflow.search_runs(
            experiment_names=[self.experiment_name],
            filter_string=filter_string,
            order_by=order_by
        )

class ModelTracker:
    """Tracks model performance and metadata."""

    def __init__(self, mlflow_manager: MLflowManager):
        """
        Initialize model tracker.

        Args:
            mlflow_manager: MLflow manager instance
        """
        self.mlflow = mlflow_manager
        self.models = {}

    def track_model(self, model_name: str, model, params: Dict[str, Any],
                   metrics: Dict[str, Any], flavor: str = "sklearn"):
        """
        Track a trained model.

        Args:
            model_name: Name of the model
            model: Trained model object
            params: Model parameters
            metrics: Model performance metrics
            flavor: MLflow flavor
        """
        with self.mlflow.start_run(run_name=f"{model_name}_training"):
            # Log parameters
            self.mlflow.log_params(params)

            # Log metrics
            self.mlflow.log_metrics(metrics)

            # Log model
            self.mlflow.log_model(model, model_name, flavor)

            # Store model info
            run_info = self.mlflow.get_run_info()
            self.models[model_name] = {
                'run_id': run_info['run_id'] if run_info else None,
                'params': params,
                'metrics': metrics,
                'flavor': flavor
            }

            logger.info(f"Tracked model: {model_name}")

    def compare_models(self, metric: str = "auc", ascending: bool = False):
        """
        Compare tracked models based on a metric.

        Args:
            metric: Metric to compare on
            ascending: Whether to sort ascending

        Returns:
            DataFrame with model comparison
        """
        if not self.models:
            return None

        comparison_data = []
        for model_name, info in self.models.items():
            if metric in info['metrics']:
                comparison_data.append({
                    'model': model_name,
                    'metric_value': info['metrics'][metric],
                    'run_id': info['run_id']
                })

        if comparison_data:
            import pandas as pd
            df = pd.DataFrame(comparison_data)
            return df.sort_values('metric_value', ascending=ascending)

        return None