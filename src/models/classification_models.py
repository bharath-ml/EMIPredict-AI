"""
Professional classification models for EMI eligibility prediction
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Tuple
import logging
import optuna
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model training"""
    random_state: int = 42
    n_trials: int = 20  # for hyperparameter optimization
    cv_folds: int = 5

class ClassificationModelTrainer:
    """Professional classification model training pipeline"""
    
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def get_models(self) -> Dict[str, Any]:
        """Initialize all classification models"""
        return {
            'Logistic Regression': LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=self.config.random_state,
                n_estimators=100,
                class_weight='balanced'
            ),
            'XGBoost': XGBClassifier(
                random_state=self.config.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.config.random_state
            ),
            'SVM': SVC(
                random_state=self.config.random_state,
                probability=True,
                class_weight='balanced'
            )
        }
    
    def optimize_hyperparameters(self, model_name: str, X_train, y_train, X_val, y_val) -> Dict:
        """Optimize hyperparameters using Optuna"""
        logger.info(f"Optimizing hyperparameters for {model_name}")
        
        def objective(trial):
            params = {}
            
            if model_name == 'Random Forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
                model = RandomForestClassifier(**params, random_state=self.config.random_state)
                
            elif model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
                model = XGBClassifier(**params, random_state=self.config.random_state, eval_metric='logloss')
                
            elif model_name == 'Gradient Boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0)
                }
                model = GradientBoostingClassifier(**params, random_state=self.config.random_state)
                
            else:
                return 0.0  # Skip optimization for other models
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            return f1
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=False)
        
        logger.info(f"Best {model_name} parameters: {study.best_params}")
        return study.best_params
    
    def train_and_evaluate(self, model_name: str, model, X_train, y_train, X_val, y_val, X_test, y_test) -> Dict:
        """Train and evaluate a single model"""
        logger.info(f"Training {model_name}")
        
        with mlflow.start_run(nested=True):
            # Log model parameters
            mlflow.set_tag("model_type", "classification")
            mlflow.set_tag("model_name", model_name)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            
            # Calculate probabilities for AUC
            y_train_proba = model.predict_proba(X_train) if hasattr(model, 'predict_proba') else None
            y_val_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
            y_test_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
                'val_f1': f1_score(y_val, y_val_pred, average='weighted'),
                'test_f1': f1_score(y_test, y_test_pred, average='weighted'),
                'train_precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
                'val_precision': precision_score(y_val, y_val_pred, average='weighted', zero_division=0),
                'test_precision': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                'train_recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
                'val_recall': recall_score(y_val, y_val_pred, average='weighted', zero_division=0),
                'test_recall': recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            }
            
            # Add ROC-AUC if available (for binary classification)
            if y_train_proba is not None and len(np.unique(y_train)) == 2:
                metrics['train_roc_auc'] = roc_auc_score(y_train, y_train_proba[:, 1])
                metrics['val_roc_auc'] = roc_auc_score(y_val, y_val_proba[:, 1])
                metrics['test_roc_auc'] = roc_auc_score(y_test, y_test_proba[:, 1])
            
            # Log parameters and metrics to MLflow
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, f"classification_{model_name.lower().replace(' ', '_')}")
            
            logger.info(f"{model_name} - Test Accuracy: {metrics['test_accuracy']:.4f}, Test F1: {metrics['test_f1']:.4f}")
            
            return {
                'model': model,
                'metrics': metrics,
                'predictions': {
                    'train': y_train_pred,
                    'val': y_val_pred,
                    'test': y_test_pred
                }
            }
    
    def run_pipeline(self, X_train, y_train, X_val, y_val, X_test, y_test) -> Dict:
        """Run complete training pipeline"""
        logger.info("Starting classification model training pipeline")
        
        results = {}
        models = self.get_models()
        
        for model_name, model in models.items():
            try:
                # Optimize hyperparameters for complex models
                if model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
                    best_params = self.optimize_hyperparameters(model_name, X_train, y_train, X_val, y_val)
                    model.set_params(**best_params)
                
                # Train and evaluate
                results[model_name] = self.train_and_evaluate(
                    model_name, model, X_train, y_train, X_val, y_val, X_test, y_test
                )
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Select best model based on validation F1 score
        best_model_name = max(results.keys(), 
                             key=lambda x: results[x]['metrics']['val_f1'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        logger.info(f"Best classification model: {best_model_name} with val F1: {results[best_model_name]['metrics']['val_f1']:.4f}")
        
        return results