"""
Professional data preprocessing module with comprehensive data validation
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
import logging
from dataclasses import dataclass
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data preprocessing"""
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    target_col_class: str = 'emi_eligibility'
    target_col_reg: str = 'max_monthly_emi'
    
class DataPreprocessor:
    """Professional data preprocessing pipeline"""
    
    def __init__(self, config: DataConfig = DataConfig()):
        self.config = config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def load_and_validate(self, filepath: str) -> pd.DataFrame:
        """Load and validate dataset"""
        logger.info(f"Loading dataset from {filepath}")
        df = pd.read_csv(filepath)
        
        # Basic validation
        assert df.shape[0] > 0, "Empty dataset"
        assert self.config.target_col_class in df.columns, f"Missing target column: {self.config.target_col_class}"
        assert self.config.target_col_reg in df.columns, f"Missing target column: {self.config.target_col_reg}"
        
        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data cleaning"""
        logger.info("Starting data cleaning process")
        
        # Create copy to avoid warnings
        df_clean = df.copy()
        
        # Handle missing values
        missing_before = df_clean.isnull().sum().sum()
        
        # Numerical columns imputation
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numerical_cols] = self.imputer.fit_transform(df_clean[numerical_cols])
        
        # Categorical columns imputation
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df_clean)
        
        # Handle outliers in numerical columns
        for col in numerical_cols:
            if col not in [self.config.target_col_reg, 'credit_score']:  # Skip target and critical columns
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        missing_after = df_clean.isnull().sum().sum()
        
        logger.info(f"Cleaning completed: Removed {duplicates_removed} duplicates, "
                   f"Missing values: {missing_before} -> {missing_after}")
        return df_clean
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        logger.info("Encoding categorical variables")
        df_encoded = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols 
                          if col not in [self.config.target_col_class]]
        
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            df_encoded[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        # Encode target variable if it's categorical
        if df[self.config.target_col_class].dtype == 'object':
            self.label_encoders['target'] = LabelEncoder()
            df_encoded[self.config.target_col_class] = self.label_encoders['target'].fit_transform(
                df[self.config.target_col_class]
            )
        
        logger.info(f"Encoded {len(categorical_cols)} categorical columns")
        return df_encoded
    
    def create_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train, validation, test splits"""
        logger.info("Creating train/val/test splits")
        
        # Separate features and targets
        feature_cols = [col for col in df.columns 
                       if col not in [self.config.target_col_class, self.config.target_col_reg]]
        
        X = df[feature_cols]
        y_class = df[self.config.target_col_class]
        y_reg = df[self.config.target_col_reg]
        
        # First split: train+val vs test
        X_temp, X_test, y_class_temp, y_class_test, y_reg_temp, y_reg_test = train_test_split(
            X, y_class, y_reg, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=y_class
        )
        
        # Second split: train vs val
        val_ratio = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val = train_test_split(
            X_temp, y_class_temp, y_reg_temp,
            test_size=val_ratio,
            random_state=self.config.random_state,
            stratify=y_class_temp
        )
        
        logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return (X_train, X_val, X_test), (y_class_train, y_class_val, y_class_test), (y_reg_train, y_reg_val, y_reg_test)
    
    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Scale numerical features"""
        logger.info("Scaling numerical features")
        
        # Fit on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def run_pipeline(self, filepath: str) -> Dict[str, Any]:
        """Run complete preprocessing pipeline"""
        logger.info("Starting complete preprocessing pipeline")
        
        # Load and validate
        df = self.load_and_validate(filepath)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Feature engineering (will be done separately)
        
        # Encode categorical
        df_encoded = self.encode_categorical(df_clean)
        
        # Create splits
        (X_train, X_val, X_test), (y_class_train, y_class_val, y_class_test), (y_reg_train, y_reg_val, y_reg_test) = \
            self.create_splits(df_encoded)
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(X_train, X_val, X_test)
        
        logger.info("Preprocessing pipeline completed successfully")
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_class_train': y_class_train,
            'y_class_val': y_class_val,
            'y_class_test': y_class_test,
            'y_reg_train': y_reg_train,
            'y_reg_val': y_reg_val,
            'y_reg_test': y_reg_test,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': X_train.columns.tolist()
        }