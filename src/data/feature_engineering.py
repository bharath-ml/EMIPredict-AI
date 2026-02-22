"""
Advanced feature engineering module for financial risk assessment
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Professional feature engineering pipeline"""
    
    def __init__(self):
        self.engineered_features = []
        
    def create_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive financial ratios"""
        logger.info("Creating financial ratios")
        df_feat = df.copy()
        
        # Debt-to-Income Ratio (DTI)
        if all(col in df.columns for col in ['current_emi_amount', 'monthly_salary']):
            df_feat['dti_ratio'] = (df['current_emi_amount'] / df['monthly_salary'].clip(lower=1)) * 100
            self.engineered_features.append('dti_ratio')
        
        # Expense-to-Income Ratio
        expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 'travel_expenses', 
                       'groceries_utilities', 'other_monthly_expenses']
        if all(col in df.columns for col in expense_cols + ['monthly_salary']):
            df_feat['total_monthly_expenses'] = df[expense_cols].sum(axis=1)
            df_feat['expense_income_ratio'] = (df_feat['total_monthly_expenses'] / df['monthly_salary'].clip(lower=1)) * 100
            self.engineered_features.extend(['total_monthly_expenses', 'expense_income_ratio'])
        
        # Savings Capacity
        if all(col in df.columns for col in ['monthly_salary', 'current_emi_amount']):
            df_feat['disposable_income'] = df['monthly_salary'] - df_feat.get('total_monthly_expenses', 0) - df['current_emi_amount']
            df_feat['savings_capacity'] = (df_feat['disposable_income'] / df['monthly_salary'].clip(lower=1)) * 100
            self.engineered_features.extend(['disposable_income', 'savings_capacity'])
        
        logger.info(f"Created {len(self.engineered_features)} financial ratios")
        return df_feat
    
    def create_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced risk scoring features"""
        logger.info("Creating risk scores")
        df_feat = df.copy()
        
        # Credit Score Category
        if 'credit_score' in df.columns:
            df_feat['credit_score_category'] = pd.cut(
                df['credit_score'],
                bins=[0, 579, 669, 739, 799, 850],
                labels=['Very Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
            )
            self.engineered_features.append('credit_score_category')
        
        # Employment Stability Score
        if all(col in df.columns for col in ['years_of_employment', 'employment_type']):
            emp_type_map = {'Government': 1.2, 'Private': 1.0, 'Self-employed': 0.8}
            df_feat['employment_stability'] = df['years_of_employment'] * df['employment_type'].map(emp_type_map).fillna(1.0)
            self.engineered_features.append('employment_stability')
        
        # Financial Burden Score
        if all(col in df.columns for col in ['dependents', 'existing_loans']):
            df_feat['has_dependents'] = (df['dependents'] > 0).astype(int)
            df_feat['has_existing_loans'] = (df['existing_loans'] == 'Yes').astype(int)
            df_feat['financial_burden_score'] = (
                df_feat['has_dependents'] * 0.3 + 
                df_feat['has_existing_loans'] * 0.4 + 
                (df['dependents'].clip(upper=3) / 3) * 0.3
            )
            self.engineered_features.extend(['has_dependents', 'has_existing_loans', 'financial_burden_score'])
        
        logger.info(f"Created {len(self.engineered_features)} risk scores")
        return df_feat
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables"""
        logger.info("Creating interaction features")
        df_feat = df.copy()
        
        # Age and Experience Interaction
        if all(col in df.columns for col in ['age', 'years_of_employment']):
            df_feat['experience_ratio'] = df['years_of_employment'] / df['age'].clip(lower=1)
            self.engineered_features.append('experience_ratio')
        
        # Salary and Tenure Interaction for requested loans
        if all(col in df.columns for col in ['monthly_salary', 'requested_tenure', 'requested_amount']):
            df_feat['requested_emi'] = df['requested_amount'] / df['requested_tenure'].clip(lower=1)
            df_feat['requested_emi_to_salary'] = (df_feat['requested_emi'] / df['monthly_salary'].clip(lower=1)) * 100
            self.engineered_features.extend(['requested_emi', 'requested_emi_to_salary'])
        
        # Family Size and Expenses Interaction
        if all(col in df.columns for col in ['family_size', 'total_monthly_expenses']):
            df_feat['expense_per_family_member'] = df_feat['total_monthly_expenses'] / df['family_size'].clip(lower=1)
            self.engineered_features.append('expense_per_family_member')
        
        logger.info(f"Created {len(self.engineered_features)} interaction features")
        return df_feat
    
    def create_loan_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create loan-specific features based on EMI scenarios"""
        logger.info("Creating loan-specific features")
        df_feat = df.copy()
        
        if 'emi_scenario' in df.columns and 'requested_amount' in df.columns:
            # Scenario-specific amount categories
            scenario_thresholds = {
                'E-commerce Shopping EMI': {'low': 50000, 'medium': 100000},
                'Home Appliances EMI': {'low': 100000, 'medium': 200000},
                'Vehicle EMI': {'low': 500000, 'medium': 1000000},
                'Personal Loan EMI': {'low': 300000, 'medium': 600000},
                'Education EMI': {'low': 200000, 'medium': 400000}
            }
            
            def categorize_amount(row):
                scenario = row['emi_scenario']
                amount = row['requested_amount']
                if scenario in scenario_thresholds:
                    if amount <= scenario_thresholds[scenario]['low']:
                        return 'Low'
                    elif amount <= scenario_thresholds[scenario]['medium']:
                        return 'Medium'
                    else:
                        return 'High'
                return 'Unknown'
            
            df_feat['amount_category'] = df_feat.apply(categorize_amount, axis=1)
            self.engineered_features.append('amount_category')
        
        logger.info(f"Created {len(self.engineered_features)} loan-specific features")
        return df_feat
    
    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run complete feature engineering pipeline"""
        logger.info("Starting complete feature engineering pipeline")
        
        # Apply all feature engineering steps
        df_feat = self.create_financial_ratios(df)
        df_feat = self.create_risk_scores(df_feat)
        df_feat = self.create_interaction_features(df_feat)
        df_feat = self.create_loan_specific_features(df_feat)
        
        logger.info(f"Feature engineering completed. Total features: {len(df_feat.columns)}")
        logger.info(f"New features created: {len(self.engineered_features)}")
        
        return df_feat