"""
Professional regression models for maximum EMI amount prediction
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from typing import Dict, Any
import logging
import optuna

logger = logging.getLogger(__name__)

class RegressionModelTrainer:
    """Professional regression model training pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def get_models(self) -> Dict[str, Any]:
        """Initialize all regression models"""
        return {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=self.config.random_state),
            'Lasso Regression': Lasso(random_state=self.config.random_state),
            'Random Forest': RandomForestRegressor(
                random_state=self.config.random_state,
                n_estimators=100
            ),
            'XGBoost': XGBRegressor(
                random_state=self.config.random_state,
                objective='reg:squarederror'
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                random_state=self.config.random_state
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
                model = RandomForestRegressor(**params, random_state=self.config.random_state)
                
            elif model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
                model = XGBRegressor(**params, random_state=self.config.random_state, objective='reg:squarederror')
                
            elif model_name == 'Gradient Boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0)
                }
                model = GradientBoostingRegressor(**params, random_state=self.config.random_state)
                
            else:
                return float('inf')
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            return rmse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=False)
        
        logger.info(f"Best {model_name} parameters: {study.best_params}")
        return study.best_params
    
    def train_and_evaluate(self, model_name: str, model, X_train, y_train, X_val, y_val, X_test, y_test) -> Dict:
        """Train and evaluate a single regression model"""
        logger.info(f"Training {model_name}")
        
        with mlflow.start_run(nested=True):
            # Log model parameters
            mlflow.set_tag("model_type", "regression")
            mlflow.set_tag("model_name", model_name)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'val_mae': mean_absolute_error(y_val, y_val_pred),
                'test_mae': mean_absolute_error(y_test, y_test_pred),
                'train_r2': r2_score(y_train, y_train_pred),
                'val_r2': r2_score(y_val, y_val_pred),
                'test_r2': r2_score(y_test, y_test_pred)
            }
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            def mape(y_true, y_pred):
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            metrics['train_mape'] = mape(y_train, y_train_pred)
            metrics['val_mape'] = mape(y_val, y_val_pred)
            metrics['test_mape'] = mape(y_test, y_test_pred)
            
            # Log parameters and metrics to MLflow
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, f"regression_{model_name.lower().replace(' ', '_')}")
            
            logger.info(f"{model_name} - Test RMSE: {metrics['test_rmse']:.2f}, Test R2: {metrics['test_r2']:.4f}")
            
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
        logger.info("Starting regression model training pipeline")
        
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
        
        # Select best model based on validation RMSE
        best_model_name = min(results.keys(), 
                             key=lambda x: results[x]['metrics']['val_rmse'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        logger.info(f"Best regression model: {best_model_name} with val RMSE: {results[best_model_name]['metrics']['val_rmse']:.2f}")
        
        return results