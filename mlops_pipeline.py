
#!/usr/bin/env python3
"""
Unified MLOps Pipeline Script
Combines pipeline orchestration, verification, and quick test utilities.
"""

import sys
import os
import argparse
import logging
import time
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_prep.data_prep import DataPreprocessor
from training.mlflow_config import MLflowManager
from training.model_training import ModelTrainer
from optimization.hyperparameter_optimization import HyperparameterOptimizer
from explainability.model_explainability import ModelExplainer
from industrialization.model_deployment import ProductionPipeline
from training.thresholding import optimize_threshold_by_cost

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class MLOpsPipeline:
    """Complete MLOps pipeline orchestrator."""
    def __init__(self, data_dir: str = "home-credit-default-risk-DATA",
                 experiment_name: str = "Home Credit Default Risk"):
        self.data_dir = data_dir
        self.experiment_name = experiment_name
        self.mlflow_manager = MLflowManager(experiment_name=experiment_name)
        self.data_preprocessor = DataPreprocessor(data_dir=data_dir)
        self.model_trainer = ModelTrainer(mlflow_manager=self.mlflow_manager)
        self.optimizer = HyperparameterOptimizer(mlflow_manager=self.mlflow_manager)
        self.explainer = ModelExplainer(mlflow_manager=self.mlflow_manager)
        self.production_pipeline = ProductionPipeline()
        logger.info("MLOps Pipeline initialized")

    def run_data_preparation(self, sample_size: float = 0.3):
        logger.info("Starting data preparation...")
        self.data_preprocessor.load_data()
        X, y, feature_names = self.data_preprocessor.prepare_main_dataset(sample_size=sample_size)
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_preprocessor.create_train_val_test_split(X, y)
        logger.info("Data preparation completed")
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

    def run_model_training(self, X_train, y_train, X_val, y_val):
        logger.info("Starting model training...")
        trained_models = self.model_trainer.train_all_models(X_train, y_train, X_val, y_val)
        cv_results = {}
        for model_name in ['logistic_regression', 'random_forest', 'lightgbm', 'xgboost', 'mlp']:
            cv_result = self.model_trainer.cross_validate_model(model_name, X_train, y_train)
            cv_results[model_name] = cv_result
        logger.info("Model training completed")
        return trained_models, cv_results

    def run_hyperparameter_optimization(self, X_train, y_train):
        logger.info("Starting hyperparameter optimization...")
        best_params = self.optimizer.optimize_all_models(X_train, y_train)
        logger.info("Hyperparameter optimization completed")
        return best_params

    def run_model_explainability(self, models, X_data, feature_names):
        logger.info("Starting model explainability analysis...")
        model_types = {
            'logistic_regression': 'linear',
            'random_forest': 'tree',
            'lightgbm': 'tree',
            'xgboost': 'tree',
            'mlp': 'deep'
        }
        comparison_df = self.explainer.compare_models_shap(models, X_data, model_types)
        self.explainer.create_lime_explainer(X_data, feature_names)
        logger.info("Model explainability analysis completed")
        return comparison_df

    def run_model_deployment(self, best_model, model_name, metadata):
        logger.info("Starting model deployment...")
        success = self.production_pipeline.setup_production_model(best_model, metadata)
        if success:
            logger.info("Model deployment completed successfully")
            return True
        else:
            logger.error("Model deployment failed")
            return False

    def select_best_model(self, cv_results, trained_models):
        logger.info("Selecting best model based on CV performance...")
        best_model_name = max(cv_results.keys(), key=lambda x: cv_results[x]['mean_auc'])
        best_model = trained_models[best_model_name]
        best_score = cv_results[best_model_name]['mean_auc']
        logger.info(f"Best model: {best_model_name} with CV AUC: {best_score:.4f}")
        return best_model_name, best_model, best_score

    def run_complete_pipeline(self):
        logger.info("Starting complete MLOps pipeline...")
        try:
            X_train, X_val, X_test, y_train, y_val, y_test, feature_names = self.run_data_preparation(sample_size=0.3)

            if self.mlflow_manager:
                with self.mlflow_manager.start_run(run_name="Full Pipeline"):
                    # Log datasets (small samples for artifact + dataset logging)
                    try:
                        import pandas as pd
                        train_df = pd.concat([X_train, y_train.rename('TARGET')], axis=1)
                        val_df = pd.concat([X_val, y_val.rename('TARGET')], axis=1)
                        test_df = pd.concat([X_test, y_test.rename('TARGET')], axis=1)
                        self.mlflow_manager.log_input_dataset(train_df.head(5000), name="train", context="training")
                        self.mlflow_manager.log_input_dataset(val_df.head(2000), name="validation", context="validation")
                        self.mlflow_manager.log_input_dataset(test_df.head(2000), name="test", context="test")
                    except Exception as e:
                        logger.warning(f"Failed to log datasets: {e}")

                    # Tag run with context
                    import mlflow
                    mlflow.set_tags({
                        'pipeline': 'full',
                        'dataset': 'home_credit',
                        'objective': 'default_risk_classification',
                    })

                    trained_models, cv_results = self.run_model_training(X_train, y_train, X_val, y_val)
                    best_params = self.run_hyperparameter_optimization(X_train, y_train)
            else:
                trained_models, cv_results = self.run_model_training(X_train, y_train, X_val, y_val)
                best_params = self.run_hyperparameter_optimization(X_train, y_train)
            self.run_model_explainability(trained_models, X_train, feature_names)
            best_model_name, best_model, best_score = self.select_best_model(cv_results, trained_models)

            # Business threshold optimization (FN cost > FP cost)
            y_val_proba = self.model_trainer.get_model_predictions(best_model_name, X_val)
            thr_metrics = optimize_threshold_by_cost(y_val, y_val_proba, fn_cost=5.0, fp_cost=1.0)

            # Log thresholding decision
            try:
                with self.mlflow_manager.start_run(run_name="Threshold Optimization", nested=True):
                    self.mlflow_manager.log_params({'fn_cost': 5.0, 'fp_cost': 1.0, 'selected_model': best_model_name})
                    self.mlflow_manager.log_metrics({
                        'opt_threshold': thr_metrics['threshold'],
                        'opt_cost': thr_metrics['cost'],
                        'opt_precision': thr_metrics['precision'],
                        'opt_recall': thr_metrics['recall'],
                        'opt_f1': thr_metrics['f1'],
                        'opt_pr_auc': thr_metrics['pr_auc']
                    })
            except Exception:
                pass
            metadata = {
                'model_name': best_model_name,
                'cv_auc': best_score,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'test_samples': len(X_test),
                'features': len(feature_names),
                'threshold': thr_metrics['threshold'],
                'cost_fn_weight': 5.0,
                'cost_fp_weight': 1.0
            }
            deployment_success = self.run_model_deployment(best_model, best_model_name, metadata)
            # Export concise summary JSON and log as MLflow artifact
            try:
                summary = {
                    'best_model_name': best_model_name,
                    'cv_auc': float(best_score),
                    'threshold': float(thr_metrics['threshold']),
                    'costs': {
                        'fn_cost': 5.0,
                        'fp_cost': 1.0,
                        'opt_cost': float(thr_metrics['cost'])
                    },
                    'training_samples': int(len(X_train)),
                    'validation_samples': int(len(X_val)),
                    'features': int(len(feature_names)),
                    'deployment_status': 'Success' if deployment_success else 'Failed'
                }
                out_dir = Path('model_registry')
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / 'best_model_summary.json'
                with open(out_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                # Log to MLflow as a nested run
                if self.mlflow_manager:
                    with self.mlflow_manager.start_run(run_name="Pipeline Summary", nested=True):
                        self.mlflow_manager.log_artifact(str(out_path), artifact_path="reports")
            except Exception as e:
                logger.warning(f"Failed to export/log pipeline summary: {e}")
            logger.info("="*50)
            logger.info("MLOPS PIPELINE SUMMARY")
            logger.info("="*50)
            logger.info(f"Best Model: {best_model_name}")
            logger.info(f"CV AUC Score: {best_score:.4f}")
            logger.info(f"Training Samples: {len(X_train)}")
            logger.info(f"Features: {len(feature_names)}")
            logger.info(f"Deployment Status: {'Success' if deployment_success else 'Failed'}")
            logger.info("="*50)
            return {
                'best_model_name': best_model_name,
                'best_model': best_model,
                'cv_score': best_score,
                'deployment_success': deployment_success,
                'cv_results': cv_results,
                'best_params': best_params
            }
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

# Quick test utility
def test_optimized_pipeline():
    """Test the optimized pipeline with reduced parameters."""
    start_time = time.time()
    logger.info("Starting optimized pipeline test...")
    data_preprocessor = DataPreprocessor(data_dir="home-credit-default-risk-DATA")
    mlflow_manager = MLflowManager(experiment_name="Optimized Pipeline Test")
    model_trainer = ModelTrainer(mlflow_manager=mlflow_manager, cv_folds=3)
    data_preprocessor.load_data()
    X, y, feature_names = data_preprocessor.prepare_main_dataset(sample_size=0.3)
    X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessor.create_train_val_test_split(X, y, test_size=0.2, val_size=0.2)
    trained_models = model_trainer.train_all_models(X_train, y_train, X_val, y_val)
    model_trainer.cross_validate_model('lightgbm', X_train, y_train)
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Training completed successfully with {len(trained_models)} models")
    if total_time < 3600:
        logger.info("SUCCESS: Pipeline completed in under 1 hour!")
        return True
    else:
        logger.warning(f"Pipeline took {total_time:.2f} seconds (>1 hour)")
        return False

# Verification utility
class MLOpsPipelineVerifier:
    """Comprehensive verifier for the MLOps pipeline implementation."""
    def __init__(self):
        self.mlflow_manager = None
        self.data_preprocessor = None
        self.model_trainer = None
        self.optimizer = None
        self.results = {}
    def run_full_verification(self):
        logger.info("Starting full pipeline verification...")
        # Example: just run the main pipeline and check for errors
        try:
            pipeline = MLOpsPipeline()
            results = pipeline.run_complete_pipeline()
            logger.info("Pipeline ran successfully.")
            self.results = results
            return {'status': 'SUCCESS', 'results': results}
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}

def main():
    """Main function to run the unified MLOps pipeline or explainability-only stage."""

    parser = argparse.ArgumentParser(description="Unified MLOps pipeline")
    subparsers = parser.add_subparsers(dest="cmd", help="Command to run")

    # Full pipeline (default)
    full_p = subparsers.add_parser("full", help="Run the full pipeline (default)")
    full_p.add_argument("--data-dir", default="home-credit-default-risk-DATA", help="Path to data directory")

    # Explainability-only
    exp_p = subparsers.add_parser("explain", help="Run only explainability: prep + (subset) train + SHAP/LIME")
    exp_p.add_argument("--data-dir", default="home-credit-default-risk-DATA", help="Path to data directory")
    exp_p.add_argument("--sample-size", type=float, default=0.1, help="Fraction of data to sample for speed (0-1]")
    exp_p.add_argument(
        "--models",
        default=None,
        help="Comma-separated subset to train: logistic_regression,random_forest,lightgbm,xgboost,mlp"
    )
    exp_p.add_argument("--explain-sample-size", type=int, default=500, help="Rows to use for SHAP/LIME (smaller = faster)")

    args = parser.parse_args()

    # Default to full if no subcommand provided
    cmd = args.cmd or "full"

    if cmd == "explain":
        data_dir = args.data_dir
        if not os.path.exists(data_dir):
            logger.error(f"Data directory {data_dir} not found!")
            sys.exit(1)

        pipeline = MLOpsPipeline(data_dir=data_dir)
        # Data prep (reduced sample by default for speed)
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = pipeline.run_data_preparation(
            sample_size=args.sample_size
        )

        # Train subset or all
        models = {}
        if args.models:
            requested = [m.strip() for m in args.models.split(',') if m.strip()]
            logger.info(f"Training subset for explainability: {requested}")
            # Use direct train_* methods to avoid unnecessary CV
            mt = pipeline.model_trainer
            for name in requested:
                if name == 'logistic_regression':
                    models[name] = mt.train_logistic_regression(X_train, y_train, X_val, y_val, start_mlflow_run=False)
                elif name == 'random_forest':
                    models[name] = mt.train_random_forest(X_train, y_train, X_val, y_val, start_mlflow_run=False)
                elif name == 'lightgbm':
                    models[name] = mt.train_lightgbm(X_train, y_train, X_val, y_val, start_mlflow_run=False)
                elif name == 'xgboost':
                    models[name] = mt.train_xgboost(X_train, y_train, X_val, y_val, start_mlflow_run=False)
                elif name == 'mlp':
                    models[name] = mt.train_mlp(X_train, y_train, X_val, y_val, start_mlflow_run=False)
                else:
                    logger.warning(f"Unknown model '{name}' requested; skipping.")
        else:
            models, _ = pipeline.run_model_training(X_train, y_train, X_val, y_val)

        # Run explainability on sampled training features for speed
        if len(X_train) > args.explain_sample_size:
            X_for_explain = X_train.sample(args.explain_sample_size, random_state=42)
        else:
            X_for_explain = X_train
        comparison_df = pipeline.run_model_explainability(models, X_for_explain, feature_names)
        print("Explainability run complete. Check MLflow for 'SHAP Model Comparison' artifacts.")
        if comparison_df is not None:
            print("Top 10 combined features:\n", comparison_df.head(10))
        return {
            'comparison': comparison_df.head(10).to_dict() if comparison_df is not None else None
        }

    # Fallback: run full pipeline
    data_dir = args.data_dir if hasattr(args, 'data_dir') else "home-credit-default-risk-DATA"
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} not found!")
        sys.exit(1)
    pipeline = MLOpsPipeline(data_dir=data_dir)
    results = pipeline.run_complete_pipeline()
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best Model: {results['best_model_name']}")
    print(f"Deployment: {'Successful' if results['deployment_success'] else 'Failed'}")
    print("="*60)
    return results

if __name__ == "__main__":
    main()