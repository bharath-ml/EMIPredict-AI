"""
Utility Helper Functions
Common utility functions used across the project
"""
import pandas as pd
import numpy as np
import json
import yaml
import pickle
import joblib
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import re
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration management class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        ext = Path(path).suffix.lower()
        
        if ext == '.json':
            with open(path, 'r') as f:
                self.config = json.load(f)
        elif ext in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")
        
        return self.config
    
    def save_config(self, path: str):
        """Save configuration to file"""
        ext = Path(path).suffix.lower()
        
        if ext == '.json':
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=2)
        elif ext in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(self.config, f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number (10 digits)"""
        pattern = r'^\d{10}$'
        return bool(re.match(pattern, phone))
    
    @staticmethod
    def validate_pan(pan: str) -> bool:
        """Validate PAN card format (India)"""
        pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
        return bool(re.match(pattern, pan))
    
    @staticmethod
    def validate_aadhar(aadhar: str) -> bool:
        """Validate Aadhar number (12 digits)"""
        pattern = r'^\d{12}$'
        return bool(re.match(pattern, aadhar))
    
    @staticmethod
    def validate_credit_score(score: int) -> bool:
        """Validate credit score range"""
        return 300 <= score <= 850
    
    @staticmethod
    def validate_age(age: int) -> bool:
        """Validate age range"""
        return 18 <= age <= 100

class FinancialCalculations:
    """Financial calculation utilities"""
    
    @staticmethod
    def calculate_emi(principal: float, rate: float, tenure: int) -> float:
        """
        Calculate EMI using formula: P * r * (1+r)^n / ((1+r)^n - 1)
        
        Args:
            principal: Loan amount
            rate: Annual interest rate (%)
            tenure: Loan tenure in months
            
        Returns:
            Monthly EMI amount
        """
        monthly_rate = rate / (12 * 100)
        if monthly_rate == 0:
            return principal / tenure
        
        emi = principal * monthly_rate * (1 + monthly_rate)**tenure / ((1 + monthly_rate)**tenure - 1)
        return round(emi, 2)
    
    @staticmethod
    def calculate_dti(monthly_debt: float, monthly_income: float) -> float:
        """Calculate Debt-to-Income ratio"""
        if monthly_income <= 0:
            return 0
        return (monthly_debt / monthly_income) * 100
    
    @staticmethod
    def calculate_foir(monthly_obligations: float, monthly_income: float) -> float:
        """Calculate Fixed Obligation to Income Ratio"""
        if monthly_income <= 0:
            return 0
        return (monthly_obligations / monthly_income) * 100
    
    @staticmethod
    def calculate_affordability(monthly_income: float, monthly_expenses: float,
                               existing_emi: float = 0) -> float:
        """Calculate affordable EMI amount"""
        disposable = monthly_income - monthly_expenses - existing_emi
        return max(0, disposable * 0.4)  # 40% of disposable income
    
    @staticmethod
    def calculate_interest(principal: float, rate: float, tenure: int) -> float:
        """Calculate total interest payable"""
        emi = FinancialCalculations.calculate_emi(principal, rate, tenure)
        total_payment = emi * tenure
        return total_payment - principal

class DateTimeUtils:
    """Date and time utilities"""
    
    @staticmethod
    def get_age_from_dob(dob: Union[str, datetime]) -> int:
        """Calculate age from date of birth"""
        if isinstance(dob, str):
            dob = datetime.strptime(dob, '%Y-%m-%d')
        
        today = datetime.now()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age
    
    @staticmethod
    def get_fiscal_year(date: Union[str, datetime] = None) -> str:
        """Get fiscal year (India: Apr-Mar)"""
        if date is None:
            date = datetime.now()
        elif isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        if date.month >= 4:
            return f"FY{date.year}-{date.year+1}"
        else:
            return f"FY{date.year-1}-{date.year}"
    
    @staticmethod
    def get_quarter(date: Union[str, datetime] = None) -> int:
        """Get quarter of the year (1-4)"""
        if date is None:
            date = datetime.now()
        elif isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        return (date.month - 1) // 3 + 1
    
    @staticmethod
    def date_range(start_date: str, end_date: str, freq: str = 'D'):
        """Generate date range"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if freq == 'D':
            delta = timedelta(days=1)
        elif freq == 'W':
            delta = timedelta(weeks=1)
        elif freq == 'M':
            delta = timedelta(days=30)
        else:
            raise ValueError(f"Unsupported frequency: {freq}")
        
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += delta
        
        return dates

class FileUtils:
    """File handling utilities"""
    
    @staticmethod
    def ensure_dir(path: str):
        """Ensure directory exists"""
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def get_file_size(path: str) -> str:
        """Get human-readable file size"""
        size = os.path.getsize(path)
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        
        return f"{size:.2f} TB"
    
    @staticmethod
    def safe_filename(filename: str) -> str:
        """Create safe filename (remove special characters)"""
        filename = re.sub(r'[^\w\s-]', '', filename)
        filename = re.sub(r'[-\s]+', '-', filename)
        return filename.strip().lower()
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Get file extension"""
        return Path(filename).suffix.lower()

class HashUtils:
    """Hashing utilities"""
    
    @staticmethod
    def hash_string(text: str, algorithm: str = 'sha256') -> str:
        """Hash a string"""
        hasher = hashlib.new(algorithm)
        hasher.update(text.encode('utf-8'))
        return hasher.hexdigest()
    
    @staticmethod
    def hash_file(filepath: str, algorithm: str = 'sha256', chunksize: int = 8192) -> str:
        """Hash a file"""
        hasher = hashlib.new(algorithm)
        
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunksize):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    @staticmethod
    def generate_id(prefix: str = '', length: int = 8) -> str:
        """Generate random ID"""
        import random
        import string
        
        chars = string.ascii_uppercase + string.digits
        random_id = ''.join(random.choice(chars) for _ in range(length))
        
        if prefix:
            return f"{prefix}_{random_id}"
        return random_id

class DataFormatter:
    """Data formatting utilities"""
    
    @staticmethod
    def format_currency(amount: float, symbol: str = '₹') -> str:
        """Format currency amount"""
        return f"{symbol}{amount:,.2f}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """Format percentage"""
        return f"{value:.{decimals}f}%"
    
    @staticmethod
    def format_large_number(num: float) -> str:
        """Format large numbers (K, M, B)"""
        if num >= 1e9:
            return f"{num/1e9:.1f}B"
        elif num >= 1e6:
            return f"{num/1e6:.1f}M"
        elif num >= 1e3:
            return f"{num/1e3:.1f}K"
        else:
            return str(num)
    
    @staticmethod
    def mask_string(text: str, show_first: int = 4, show_last: int = 4,
                   mask_char: str = '*') -> str:
        """Mask a string (e.g., for PII)"""
        if len(text) <= show_first + show_last:
            return text
        
        visible = text[:show_first] + mask_char * (len(text) - show_first - show_last) + text[-show_last:]
        return visible

class ModelUtils:
    """Model utility functions"""
    
    @staticmethod
    def save_model(model, path: str, method: str = 'joblib'):
        """Save model to disk"""
        FileUtils.ensure_dir(os.path.dirname(path))
        
        if method == 'joblib':
            joblib.dump(model, path)
        elif method == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Unsupported save method: {method}")
        
        logger.info(f"Model saved to {path}")
    
    @staticmethod
    def load_model(path: str, method: str = 'joblib'):
        """Load model from disk"""
        if method == 'joblib':
            model = joblib.load(path)
        elif method == 'pickle':
            with open(path, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(f"Unsupported load method: {method}")
        
        logger.info(f"Model loaded from {path}")
        return model
    
    @staticmethod
    def get_model_size(model) -> str:
        """Get model size in memory"""
        import sys
        size = sys.getsizeof(pickle.dumps(model))
        return FileUtils.get_file_size.__func__(size)  # pylint: disable=no-member

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        FileUtils.ensure_dir(os.path.dirname(log_file))
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )