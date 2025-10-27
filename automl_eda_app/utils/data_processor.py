import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Utility class for data processing tasks."""
    
    def load_data(self, uploaded_file) -> pd.DataFrame:
        """Load data from uploaded file."""
        try:
            logger.info(f"Loading data from {uploaded_file.name}")
            if uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            logger.info(f"Successfully loaded dataset with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error reading dataset: {e}")
            raise ValueError(f"Error reading dataset: {e}")
    
    def get_data_info(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, int]:
        """Get basic information about the dataset."""
        data_types = df.dtypes.value_counts()
        missing_values = df.isnull().sum()
        total_missing = missing_values.sum()
        return data_types, missing_values, total_missing
    
    def clean_data(self, df: pd.DataFrame, target: str, 
                   drop_na: bool = True, impute_strategy: Optional[str] = None) -> pd.DataFrame:
        """Clean the dataset based on specified parameters."""
        df_clean = df.copy()
        
        # Handle missing values in target column
        df_clean = df_clean.dropna(subset=[target])
        
        if drop_na:
            # Drop rows with any missing values
            df_clean = df_clean.dropna(axis=0, how='any')
        elif impute_strategy:
            # Impute missing values based on strategy
            if impute_strategy == "mean":
                df_clean = df_clean.fillna(df_clean.mean())
            elif impute_strategy == "median":
                df_clean = df_clean.fillna(df_clean.median())
            elif impute_strategy == "mode":
                df_clean = df_clean.fillna(df_clean.mode().iloc[0])
        
        return df_clean
    
    def encode_categoricals(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, Any]:
        """Encode categorical variables and separate features and target."""
        X = df.drop(columns=[target])
        y = df[target]
        
        # One-hot encode categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        return X, y