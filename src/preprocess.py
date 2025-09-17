import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

from config import FEATURE_CONFIG, MODEL_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []
        
    def clean_data(self, df):
        df_clean = df.copy()
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        numerical_cols = FEATURE_CONFIG['numerical_features']
        missing_numerical = df_clean[numerical_cols].isnull().sum()
        # Median imputer strategy for skewed data
        if missing_numerical.sum() > 0:
            df_clean[numerical_cols] = self.imputer.fit_transform(df_clean[numerical_cols])
        
        categorical_cols = FEATURE_CONFIG['ordinal_features']
        missing_categorical = df_clean[categorical_cols].isnull().sum()
        if missing_categorical.sum() > 0:
            df_clean[categorical_cols] = df_clean[categorical_cols].fillna('Unknown')
     
        return df_clean
    
    def encode_features(self, df):
        preprocessor = make_column_transformer(
        (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=int), 
            FEATURE_CONFIG['ordinal_features']),
        (StandardScaler(), FEATURE_CONFIG['numerical_features']),
        remainder='drop' 
        )
        
        pipe = make_pipeline(
            preprocessor, 
            RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1)
        )
        
        # save pipeline
        pipe_path = "models/pipe.pkl"
        os.makedirs(os.path.dirname(pipe_path), exist_ok=True)
        pickle.dump(pipe, open(pipe_path, "wb"))
        return None

    
    def split_data(self, X: pd.DataFrame, y: pd.Series):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2,
            random_state=123
        )
        logger.info("Split data into train/test sets")
        return X_train, X_test, y_train, y_test
    


def main():
    from extract import DataExtractor
    try:
        extractor = DataExtractor()
        df = extractor.load_from_database()
        
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(df)
        X = df_clean.drop(columns = FEATURE_CONFIG['target_column'])
        y = df_clean[FEATURE_CONFIG['target_column']]
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        
        preprocessor.encode_features(X_train)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
