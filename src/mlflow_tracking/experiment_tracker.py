"""
Professional MLflow experiment tracking module
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """Professional MLflow experiment tracking"""
    
    def __init__(self, experiment_name: str = "EMIPredict_AI", tracking_uri: str = "./mlruns"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.client = None
        self.experiment_id = None
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
    def setup_experiment(self):
        """Setup MLflow experiment"""
        # Create experiment if it doesn't exist
        try:
            self.experiment_id = mlflow.create_experiment(
                name=self.experiment_name,
                tags={"project": "EMIPredict_AI", "domain": "FinTech"}
            )
        except:
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
        
        mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient()
        
        logger.info(f"Experiment setup complete. ID: {self.experiment_id}")
        
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information"""
        with mlflow.start_run(run_name="dataset_info"):
            mlflow.set_tag("run_type", "dataset_analysis")
            
            # Log dataset statistics
            for key, value in dataset_info.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"dataset_{key}", value)
                elif isinstance(value, str):
                    mlflow.set_tag(f"dataset_{key}", value)
            
            # Log feature list
            if 'features' in dataset_info:
                mlflow.set_tag("num_features", len(dataset_info['features']))
                
    def log_feature_importance(self, feature_names: List[str], importance_values: np.ndarray, model_name: str):
        """Log feature importance"""
        with mlflow.start_run(run_name=f"feature_importance_{model_name}"):
            mlflow.set_tag("run_type", "feature_importance")
            mlflow.set_tag("model_name", model_name)
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=False)
            
            # Log top features
            for i, row in importance_df.head(10).iterrows():
                mlflow.log_metric(f"feature_imp_{row['feature']}", row['importance'])
            
            # Save importance as artifact
            importance_df.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
            os.remove("feature_importance.csv")
            
    def log_model_to_registry(self, model, model_name: str, metrics: Dict[str, float], stage: str = "Staging"):
        """Log model to MLflow Model Registry"""
        with mlflow.start_run(run_name=f"model_registry_{model_name}"):
            mlflow.set_tag("run_type", "model_registry")
            mlflow.set_tag("model_name", model_name)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            model_info = mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"EMIPredict_{model_name}"
            )
            
            # Transition model stage
            if self.client:
                model_version = self.client.get_latest_versions(f"EMIPredict_{model_name}", stages=["None"])[0].version
                self.client.transition_model_version_stage(
                    name=f"EMIPredict_{model_name}",
                    version=model_version,
                    stage=stage
                )
            
            logger.info(f"Model {model_name} logged to registry with stage: {stage}")
            return model_info
            
    def compare_models(self, model_results: Dict[str, Dict], model_type: str = "classification") -> pd.DataFrame:
        """Compare multiple models and return comparison dataframe"""
        comparison_data = []
        
        for model_name, results in model_results.items():
            metrics = results['metrics']
            row = {
                'model_name': model_name,
                'model_type': model_type
            }
            
            # Add relevant metrics
            if model_type == "classification":
                row.update({
                    'test_accuracy': metrics.get('test_accuracy', 0),
                    'test_f1': metrics.get('test_f1', 0),
                    'test_precision': metrics.get('test_precision', 0),
                    'test_recall': metrics.get('test_recall', 0)
                })
                if 'test_roc_auc' in metrics:
                    row['test_roc_auc'] = metrics['test_roc_auc']
            else:
                row.update({
                    'test_rmse': metrics.get('test_rmse', 0),
                    'test_mae': metrics.get('test_mae', 0),
                    'test_r2': metrics.get('test_r2', 0),
                    'test_mape': metrics.get('test_mape', 0)
                })
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Log comparison
        with mlflow.start_run(run_name=f"model_comparison_{model_type}"):
            mlflow.set_tag("run_type", "model_comparison")
            mlflow.set_tag("model_type", model_type)
            
            # Log as artifact
            comparison_df.to_csv(f"model_comparison_{model_type}.csv", index=False)
            mlflow.log_artifact(f"model_comparison_{model_type}.csv")
            os.remove(f"model_comparison_{model_type}.csv")
        
        return comparison_df
    
    def get_experiment_results(self) -> pd.DataFrame:
        """Get all experiment results"""
        if not self.client:
            self.setup_experiment()
        
        # Get all runs
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        return runs
    
    def get_best_model(self, model_type: str, metric: str = "val_f1", direction: str = "maximize") -> Optional[Dict]:
        """Get best model from experiments"""
        runs = self.get_experiment_results()
        
        if runs.empty:
            return None
        
        # Filter by model type
        if model_type == "classification":
            runs = runs[runs['tags.model_type'] == 'classification']
            if metric not in runs.columns:
                metric = 'metrics.val_f1'
        else:
            runs = runs[runs['tags.model_type'] == 'regression']
            if metric not in runs.columns:
                metric = 'metrics.val_rmse'
        
        if runs.empty:
            return None
        
        # Sort based on direction
        if direction == "maximize":
            best_run = runs.loc[runs[metric].astype(float).idxmax()]
        else:
            best_run = runs.loc[runs[metric].astype(float).idxmin()]
        
        return best_run.to_dict()