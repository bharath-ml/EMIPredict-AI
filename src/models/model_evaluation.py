"""
Model Evaluation Module
Comprehensive evaluation metrics and visualization for ML models
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List, Optional
import logging
import joblib
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Professional model evaluation class"""
    
    def __init__(self, model_type: str = "classification"):
        """
        Initialize model evaluator
        
        Args:
            model_type: "classification" or "regression"
        """
        self.model_type = model_type
        self.metrics = {}
        self.predictions = {}
        self.probabilities = {}
        
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_prob: Optional[np.ndarray] = None,
                               average: str = 'weighted') -> Dict[str, float]:
        """
        Evaluate classification model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (for ROC-AUC)
            average: Averaging method for metrics
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # ROC-AUC (if probabilities provided and binary classification)
        if y_prob is not None and len(np.unique(y_true)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        return metrics
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate regression model
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        # Additional metrics
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        metrics['explained_variance'] = 1 - np.var(y_true - y_pred) / np.var(y_true)
        
        return metrics
    
    def calculate_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Calculate feature importance
        
        Args:
            model: Trained model with feature_importances_ or coef_ attribute
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            if len(importances.shape) > 1:
                importances = np.mean(importances, axis=0)
        else:
            logger.warning("Model does not have feature importance attribute")
            return pd.DataFrame()
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Add percentage
        importance_df['importance_percentage'] = (importance_df['importance'] / importance_df['importance'].sum()) * 100
        
        return importance_df
    
    def calculate_confidence_intervals(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      n_bootstrap: int = 1000) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals using bootstrap
        
        Args:
            y_true: True values
            y_pred: Predicted values
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary of metrics with confidence intervals
        """
        np.random.seed(42)
        n_samples = len(y_true)
        
        bootstrap_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        } if self.model_type == 'classification' else {
            'rmse': [],
            'mae': [],
            'r2': []
        }
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Calculate metrics
            if self.model_type == 'classification':
                bootstrap_metrics['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
                bootstrap_metrics['precision'].append(precision_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))
                bootstrap_metrics['recall'].append(recall_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))
                bootstrap_metrics['f1_score'].append(f1_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))
            else:
                bootstrap_metrics['rmse'].append(np.sqrt(mean_squared_error(y_true_boot, y_pred_boot)))
                bootstrap_metrics['mae'].append(mean_absolute_error(y_true_boot, y_pred_boot))
                bootstrap_metrics['r2'].append(r2_score(y_true_boot, y_pred_boot))
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for metric, values in bootstrap_metrics.items():
            lower = np.percentile(values, 2.5)
            upper = np.percentile(values, 97.5)
            confidence_intervals[metric] = (lower, upper)
        
        return confidence_intervals
    
    def plot_confusion_matrix(self, cm: np.ndarray, labels: List[str]) -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            labels: Class labels
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
        """
        Plot residuals for regression
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Matplotlib figure
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Actual vs Predicted
        axes[1, 1].scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Actual vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_prob: Optional[np.ndarray] = None,
                                  model=None, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_prob: Prediction probabilities
            model: Trained model
            feature_names: Feature names
            
        Returns:
            Dictionary with evaluation results
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'metrics': {},
            'confidence_intervals': {},
            'feature_importance': None,
            'sample_size': len(y_true)
        }
        
        # Calculate metrics
        if self.model_type == 'classification':
            report['metrics'] = self.evaluate_classification(y_true, y_pred, y_prob)
            report['confidence_intervals'] = self.calculate_confidence_intervals(y_true, y_pred)
        else:
            report['metrics'] = self.evaluate_regression(y_true, y_pred)
            report['confidence_intervals'] = self.calculate_confidence_intervals(y_true, y_pred)
        
        # Calculate feature importance
        if model is not None and feature_names is not None:
            importance_df = self.calculate_feature_importance(model, feature_names)
            if not importance_df.empty:
                report['feature_importance'] = importance_df.to_dict('records')
        
        return report
    
    def save_report(self, report: Dict[str, Any], filepath: str):
        """
        Save evaluation report to file
        
        Args:
            report: Evaluation report dictionary
            filepath: Path to save report
        """
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Evaluation report saved to {filepath}")
    
    def compare_models(self, models_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models_results: Dictionary of model results
            
        Returns:
            DataFrame with comparison
        """
        comparison = []
        
        for model_name, results in models_results.items():
            row = {'model': model_name}
            row.update(results.get('metrics', {}))
            comparison.append(row)
        
        comparison_df = pd.DataFrame(comparison)
        
        # Sort by primary metric
        if self.model_type == 'classification':
            if 'f1_score' in comparison_df.columns:
                comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        else:
            if 'rmse' in comparison_df.columns:
                comparison_df = comparison_df.sort_values('rmse')
        
        return comparison_df