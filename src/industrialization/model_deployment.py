"""
Industrialization module for model deployment and production.
Includes model registry, deployment scripts, and monitoring setup.
"""

import os
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import logging
import subprocess
import numpy as np

try:
    from training.mlflow_config import MLflowManager
    MLFLOW_AVAILABLE = True
except ImportError:
    MLflowManager = None
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Handles model registration and versioning."""

    def __init__(self, mlflow_manager: Optional[Any] = None,
                 registry_path: str = "./model_registry"):
        """
        Initialize model registry.

        Args:
            mlflow_manager: MLflow manager for tracking
            registry_path: Path to store registered models
        """
        self.mlflow = mlflow_manager
        self.registry_path = registry_path
        self.registered_models = {}

        # Create registry directory
        os.makedirs(registry_path, exist_ok=True)
        self._load_registry()

    def _load_registry(self):
        """Load existing registry information."""
        registry_file = os.path.join(self.registry_path, "registry.json")
        if os.path.exists(registry_file):
            try:
                with open(registry_file, 'r') as f:
                    self.registered_models = json.load(f)
                logger.info(f"Loaded registry with {len(self.registered_models)} models")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")

    def _save_registry(self):
        """Save registry information."""
        registry_file = os.path.join(self.registry_path, "registry.json")
        try:
            with open(registry_file, 'w') as f:
                json.dump(self.registered_models, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def register_model(self, model_name: str, model: Any, metadata: Dict[str, Any],
                      version: Optional[str] = None) -> Optional[str]:
        """
        Register a model in the registry.

        Args:
            model_name: Name of the model
            model: Trained model object
            metadata: Model metadata (parameters, metrics, etc.)
            version: Model version (auto-generated if None)

        Returns:
            Model version string
        """
        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v_{timestamp}"

        model_info = {
            'name': model_name,
            'version': version,
            'registered_at': datetime.now().isoformat(),
            'metadata': metadata,
            'status': 'active'
        }

        # Save model to disk
        model_path = os.path.join(self.registry_path, f"{model_name}_{version}.pkl")
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            model_info['model_path'] = model_path
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return None

        # Update registry
        if model_name not in self.registered_models:
            self.registered_models[model_name] = []

        self.registered_models[model_name].append(model_info)
        self._save_registry()

        # Log to MLflow
        if self.mlflow:
            with self.mlflow.start_run(run_name=f"Model Registration: {model_name}"):
                self.mlflow.log_params({
                    'model_name': model_name,
                    'version': version,
                    'registration_time': model_info['registered_at']
                })
                for key, value in metadata.items():
                    if isinstance(value, (int, float)):
                        self.mlflow.log_metrics({key: value})
                    else:
                        self.mlflow.log_param(key, str(value))

        logger.info(f"Registered model: {model_name} {version}")
        return version

    def get_model(self, model_name: str, version: Optional[str] = None) -> Optional[Any]:
        """
        Retrieve a model from the registry.

        Args:
            model_name: Name of the model
            version: Specific version (latest if None)

        Returns:
            Model object or None
        """
        if model_name not in self.registered_models:
            logger.error(f"Model {model_name} not found in registry")
            return None

        models = self.registered_models[model_name]

        if version is None:
            # Get latest version
            models.sort(key=lambda x: x['registered_at'], reverse=True)
            model_info = models[0]
        else:
            # Find specific version
            model_info = next((m for m in models if m['version'] == version), None)

        if model_info is None:
            logger.error(f"Version {version} not found for model {model_name}")
            return None

        model_path = model_info.get('model_path')
        if not model_path or not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Loaded model: {model_name} {model_info['version']}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def list_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all registered models.

        Returns:
            Dictionary of model names and their versions
        """
        return self.registered_models.copy()

    def promote_model(self, model_name: str, version: str, stage: str = "production"):
        """
        Promote a model to a specific stage.

        Args:
            model_name: Name of the model
            version: Model version
            stage: Target stage (staging, production, etc.)
        """
        if model_name not in self.registered_models:
            logger.error(f"Model {model_name} not found")
            return

        for model_info in self.registered_models[model_name]:
            if model_info['version'] == version:
                model_info['stage'] = stage
                model_info['promoted_at'] = datetime.now().isoformat()
                self._save_registry()

                if self.mlflow:
                    with self.mlflow.start_run(run_name=f"Model Promotion: {model_name}"):
                        self.mlflow.log_params({
                            'model_name': model_name,
                            'version': version,
                            'stage': stage,
                            'promotion_time': model_info['promoted_at']
                        })

                logger.info(f"Promoted {model_name} {version} to {stage}")
                return

        logger.error(f"Version {version} not found for model {model_name}")

class ModelDeployment:
    """Handles model deployment and serving."""

    def __init__(self, model_registry: ModelRegistry):
        """
        Initialize model deployment.

        Args:
            model_registry: Model registry instance
        """
        self.registry = model_registry
        self.deployed_models = {}

    def deploy_model(self, model_name: str, version: Optional[str] = None,
                    deployment_name: Optional[str] = None) -> bool:
        """
        Deploy a model for serving.

        Args:
            model_name: Name of the model to deploy
            version: Model version (latest if None)
            deployment_name: Name for the deployment

        Returns:
            True if deployment successful, False otherwise
        """
        model = self.registry.get_model(model_name, version)
        if model is None:
            return False

        if deployment_name is None:
            deployment_name = f"{model_name}_deployment"

        self.deployed_models[deployment_name] = {
            'model_name': model_name,
            'version': version or 'latest',
            'model': model,
            'deployed_at': datetime.now().isoformat(),
            'status': 'active'
        }

        logger.info(f"Deployed model: {deployment_name}")
        return True

    def predict(self, deployment_name: str, X: Any) -> Optional[Any]:
        """
        Make predictions using a deployed model.

        Args:
            deployment_name: Name of the deployment
            X: Input data

        Returns:
            Model predictions or None
        """
        if deployment_name not in self.deployed_models:
            logger.error(f"Deployment {deployment_name} not found")
            return None

        deployment = self.deployed_models[deployment_name]
        if deployment['status'] != 'active':
            logger.error(f"Deployment {deployment_name} is not active")
            return None

        model = deployment['model']

        try:
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)
            else:
                return model.predict(X)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None

    def list_deployments(self) -> Dict[str, Dict[str, Any]]:
        """
        List all active deployments.

        Returns:
            Dictionary of deployment information
        """
        return {name: {k: v for k, v in info.items() if k != 'model'}
                for name, info in self.deployed_models.items()}

class ModelMonitor:
    """Handles model monitoring and performance tracking."""

    def __init__(self, mlflow_manager: Optional[Any] = None,
                 monitoring_path: str = "./monitoring"):
        """
        Initialize model monitor.

        Args:
            mlflow_manager: MLflow manager for tracking
            monitoring_path: Path to store monitoring data
        """
        self.mlflow = mlflow_manager
        self.monitoring_path = monitoring_path
        self.metrics_history = {}

        os.makedirs(monitoring_path, exist_ok=True)

    def log_prediction(self, deployment_name: str, X: Any, y_pred: Any,
                      y_true: Any = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Log a prediction for monitoring.

        Args:
            deployment_name: Name of the deployment
            X: Input features
            y_pred: Model predictions
            y_true: True labels (if available)
            metadata: Additional metadata
        """
        timestamp = datetime.now().isoformat()

        log_entry = {
            'timestamp': timestamp,
            'deployment': deployment_name,
            'input_shape': X.shape if hasattr(X, 'shape') else len(X),
            'prediction_shape': y_pred.shape if hasattr(y_pred, 'shape') else len(y_pred),
            'metadata': metadata or {}
        }

        if y_true is not None:
            log_entry['true_labels'] = y_true.tolist() if hasattr(y_true, 'tolist') else y_true

        # Store in memory for quick access
        if deployment_name not in self.metrics_history:
            self.metrics_history[deployment_name] = []

        self.metrics_history[deployment_name].append(log_entry)

        # Keep only last 1000 entries in memory
        if len(self.metrics_history[deployment_name]) > 1000:
            self.metrics_history[deployment_name] = self.metrics_history[deployment_name][-1000:]

        # Log to MLflow if available
        if self.mlflow:
            with self.mlflow.start_run(run_name=f"Prediction Log: {deployment_name}"):
                self.mlflow.log_params({
                    'deployment': deployment_name,
                    'timestamp': timestamp,
                    'input_shape': str(log_entry['input_shape'])
                })

    def get_monitoring_stats(self, deployment_name: str) -> Dict[str, Any]:
        """
        Get monitoring statistics for a deployment.

        Args:
            deployment_name: Name of the deployment

        Returns:
            Dictionary with monitoring statistics
        """
        if deployment_name not in self.metrics_history:
            return {}

        logs = self.metrics_history[deployment_name]

        stats = {
            'total_predictions': len(logs),
            'time_range': {
                'start': logs[0]['timestamp'] if logs else None,
                'end': logs[-1]['timestamp'] if logs else None
            },
            'average_input_shape': None,
            'average_prediction_shape': None
        }

        # Calculate averages
        if logs:
            input_shapes = [log['input_shape'] for log in logs if 'input_shape' in log]
            pred_shapes = [log['prediction_shape'] for log in logs if 'prediction_shape' in log]

            if input_shapes:
                if isinstance(input_shapes[0], tuple):
                    # For multi-dimensional shapes
                    stats['average_input_shape'] = tuple(np.mean([s for s in input_shapes], axis=0).astype(int))
                else:
                    stats['average_input_shape'] = np.mean(input_shapes)

            if pred_shapes:
                if isinstance(pred_shapes[0], tuple):
                    stats['average_prediction_shape'] = tuple(np.mean([s for s in pred_shapes], axis=0).astype(int))
                else:
                    stats['average_prediction_shape'] = np.mean(pred_shapes)

        return stats

class ProductionPipeline:
    """Complete production pipeline for model serving."""

    def __init__(self, model_name: str = "home_credit_model"):
        """
        Initialize production pipeline.

        Args:
            model_name: Name of the model to serve
        """
        self.model_name = model_name
        self.registry = ModelRegistry()
        self.deployment = ModelDeployment(self.registry)
        self.monitor = ModelMonitor()

    def setup_production_model(self, model: Any, metadata: Dict[str, Any]) -> bool:
        """
        Set up a model for production.

        Args:
            model: Trained model
            metadata: Model metadata

        Returns:
            True if setup successful, False otherwise
        """
        # Register the model
        version = self.registry.register_model(self.model_name, model, metadata)

        if version is None:
            return False

        # Deploy the model
        success = self.deployment.deploy_model(self.model_name, version, f"{self.model_name}_prod")

        if success:
            logger.info(f"Production pipeline ready for {self.model_name}")
            return True
        else:
            logger.error(f"Failed to set up production pipeline for {self.model_name}")
            return False

    def predict(self, X: Any, metadata: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Make production predictions.

        Args:
            X: Input data
            metadata: Additional metadata for monitoring

        Returns:
            Model predictions
        """
        deployment_name = f"{self.model_name}_prod"

        predictions = self.deployment.predict(deployment_name, X)

        if predictions is not None:
            # Log the prediction
            self.monitor.log_prediction(deployment_name, X, predictions,
                                      metadata=metadata)

        return predictions

    def get_production_stats(self) -> Dict[str, Any]:
        """
        Get production statistics.

        Returns:
            Dictionary with production statistics
        """
        deployment_name = f"{self.model_name}_prod"
        return self.monitor.get_monitoring_stats(deployment_name)