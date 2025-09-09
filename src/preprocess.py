import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from .config import FEATURE_CONFIG, MODEL_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning data...")
        df_clean = df.copy()
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        numerical_cols = FEATURE_CONFIG['numerical_features']
        missing_numerical = df_clean[numerical_cols].isnull().sum()
        if missing_numerical.sum() > 0:
            logger.info(f"Missing values in numerical columns: {missing_numerical[missing_numerical > 0]}")
            df_clean[numerical_cols] = self.imputer.fit_transform(df_clean[numerical_cols])
        
        categorical_cols = FEATURE_CONFIG['categorical_features']
        missing_categorical = df_clean[categorical_cols].isnull().sum()
        if missing_categorical.sum() > 0:
            logger.info(f"Missing values in categorical columns: {missing_categorical[missing_categorical > 0]}")
            df_clean[categorical_cols] = df_clean[categorical_cols].fillna('Unknown')
        
        logger.info("Data cleaning completed")
        return df_clean
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Encoding categorical features...")
        df_encoded = df.copy()
        categorical_cols = FEATURE_CONFIG['categorical_features']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Encoded column: {col}")
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        logger.info("Scaling numerical features...")
        df_scaled = df.copy()
        numerical_cols = FEATURE_CONFIG['numerical_features']
        
        if fit:
            df_scaled[numerical_cols] = self.scaler.fit_transform(df_scaled[numerical_cols])
        else:
            df_scaled[numerical_cols] = self.scaler.transform(df_scaled[numerical_cols])
        
        logger.info("Feature scaling completed")
        return df_scaled
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating derived features...")
        df_features = df.copy()
        
        if 'age' in df_features.columns and 'tenure' in df_features.columns:
            df_features['age_tenure_ratio'] = df_features['age'] / (df_features['tenure'] + 1)
        
        if 'usage_frequency' in df_features.columns and 'support_calls' in df_features.columns:
            df_features['usage_support_ratio'] = df_features['usage_frequency'] / (df_features['support_calls'] + 1)
        
        if 'payment_delay' in df_features.columns:
            payment_risk = pd.cut(
                df_features['payment_delay'], 
                bins=[0, 7, 14, 30, 100], 
                labels=[0, 1, 2, 3],
                include_lowest=True
            )
            df_features['payment_risk'] = payment_risk.fillna(0).astype(int)
        
        if 'support_calls' in df_features.columns:
            support_risk = pd.cut(
                df_features['support_calls'], 
                bins=[0, 2, 5, 10, 100], 
                labels=[0, 1, 2, 3],
                include_lowest=True
            )
            df_features['support_risk'] = support_risk.fillna(0).astype(int)
        
        if 'total_spend' in df_features.columns:
            spending_category = pd.cut(
                df_features['total_spend'], 
                bins=[0, 200, 500, 1000, 10000], 
                labels=[0, 1, 2, 3],
                include_lowest=True
            )
            df_features['spending_category'] = spending_category.fillna(0).astype(int)
        
        logger.info(f"Created {len(df_features.columns) - len(df.columns)} derived features")
        return df_features
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        logger.info("Preparing features for modeling...")
        exclude_cols = [
            FEATURE_CONFIG['target_column'], 
            FEATURE_CONFIG['customer_id_column']
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        X = df[feature_cols].copy()
        valid_rows = X.notna().all(axis=1)
        X = X[valid_rows]
        self.valid_rows = valid_rows
        logger.info(f"Prepared {len(feature_cols)} features")
        logger.info(f"Valid rows after cleaning: {len(X)}")
        return X, feature_cols
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        logger.info("Splitting data into train/validation/test sets...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=MODEL_CONFIG['test_size'] + MODEL_CONFIG['validation_size'],
            random_state=MODEL_CONFIG['random_state'],
            stratify=y
        )
        val_ratio = MODEL_CONFIG['validation_size'] / (MODEL_CONFIG['test_size'] + MODEL_CONFIG['validation_size'])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=MODEL_CONFIG['random_state'],
            stratify=y_temp
        )
        logger.info(f"Data split completed:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Validation: {len(X_val)} samples")
        logger.info(f"  Test: {len(X_test)} samples")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_feature_importance_dataframe(self, feature_importance: np.ndarray) -> pd.DataFrame:
        if len(feature_importance) != len(self.feature_names):
            logger.warning("Feature importance length doesn't match feature names")
            return pd.DataFrame()
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        return importance_df
    
    def inverse_transform_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        df_reversed = df.copy()
        for col, le in self.label_encoders.items():
            if col in df_reversed.columns:
                df_reversed[col] = le.inverse_transform(df_reversed[col])
        return df_reversed


def main():
    from .extract import DataExtractor
    try:
        extractor = DataExtractor()
        df = extractor.load_original_data()
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(df)
        df_encoded = preprocessor.encode_categorical_features(df_clean)
        df_scaled = preprocessor.scale_numerical_features(df_encoded)
        df_features = preprocessor.create_derived_features(df_scaled)
        X, feature_cols = preprocessor.prepare_features(df_features)
        y = df_features[FEATURE_CONFIG['target_column']]
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        print("=== PREPROCESSING SUMMARY ===")
        print(f"Original shape: {df.shape}")
        print(f"Final features: {len(feature_cols)}")
        print(f"Train set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
