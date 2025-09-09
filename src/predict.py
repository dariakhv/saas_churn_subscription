import pandas as pd
import numpy as np
import logging
import pickle
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .config import MODEL_FILE, PREDICTIONS_FILE, FEATURE_CONFIG
from .extract import DataExtractor
from .preprocess import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnPredictor:
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.preprocessor = None
        self.model_path = model_path or MODEL_FILE
        self.preprocessor_path = self.model_path.parent / "preprocessor.pkl"
        self.load_model()
    
    def load_model(self) -> None:
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from {self.model_path}")
            
            if self.preprocessor_path.exists():
                with open(self.preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
                logger.info(f"Preprocessor loaded from {self.preprocessor_path}")
            else:
                logger.warning("Preprocessor file not found, initializing new preprocessor")
                self.preprocessor = DataPreprocessor()
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing new data for prediction...")
        df_clean = self.preprocessor.clean_data(df)
        df_encoded = self.preprocessor.encode_categorical_features(df_clean)
        df_scaled = self.preprocessor.scale_numerical_features(df_encoded, fit=False)
        df_features = self.preprocessor.create_derived_features(df_scaled)
        X, feature_cols = self.preprocessor.prepare_features(df_features)
        logger.info(f"Preprocessed data shape: {X.shape}")
        return X, feature_cols
    
    def predict_churn(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Making churn predictions...")
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None
        logger.info(f"Predictions completed for {len(X)} samples")
        return y_pred, y_pred_proba
    
    def create_prediction_dataframe(self, df: pd.DataFrame, y_pred: np.ndarray, 
                                  y_pred_proba: np.ndarray, 
                                  customer_ids: pd.Series) -> pd.DataFrame:
        prediction_df = df.copy()
        prediction_df['Churn_Prediction'] = y_pred
        prediction_df['Churn_Probability'] = y_pred_proba if y_pred_proba is not None else np.nan
        
        if customer_ids is not None:
            prediction_df['CustomerID'] = customer_ids
        
        if y_pred_proba is not None:
            prediction_df['Risk_Category'] = pd.cut(
                y_pred_proba,
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
        return prediction_df
    
    def get_high_risk_customers(self, prediction_df: pd.DataFrame, 
                               threshold: float = 0.7) -> pd.DataFrame:
        if 'Churn_Probability' not in prediction_df.columns:
            logger.warning("Churn probability not available")
            return pd.DataFrame()
        
        high_risk = prediction_df[prediction_df['Churn_Probability'] >= threshold].copy()
        high_risk = high_risk.sort_values('Churn_Probability', ascending=False)
        logger.info(f"Found {len(high_risk)} high-risk customers (threshold: {threshold})")
        return high_risk
    
    def save_predictions(self, prediction_df: pd.DataFrame, 
                        filepath: Optional[Path] = None) -> None:
        filepath = filepath or PREDICTIONS_FILE
        try:
            prediction_df.to_csv(filepath, index=False)
            logger.info(f"Predictions saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
            raise
    
    def generate_prediction_summary(self, prediction_df: pd.DataFrame) -> Dict:
        summary = {
            'total_customers': len(prediction_df),
            'predicted_churn': int(prediction_df['Churn_Prediction'].sum()),
            'churn_rate': float(prediction_df['Churn_Prediction'].mean()),
            'avg_churn_probability': float(prediction_df['Churn_Probability'].mean())
        }
        
        if 'Risk_Category' in prediction_df.columns:
            risk_distribution = prediction_df['Risk_Category'].value_counts().to_dict()
            summary['risk_distribution'] = risk_distribution
        
        if 'Churn_Probability' in prediction_df.columns:
            high_risk = prediction_df[prediction_df['Churn_Probability'] >= 0.7]
            if len(high_risk) > 0:
                summary['high_risk_customers'] = len(high_risk)
                summary['high_risk_avg_probability'] = float(high_risk['Churn_Probability'].mean())
        
        return summary
    
    def predict_batch(self, data_path: Optional[Path] = None, 
                     save_predictions: bool = True) -> pd.DataFrame:
        try:
            if data_path:
                logger.info(f"Loading data from {data_path}")
                df = pd.read_csv(data_path)
            else:
                logger.info("Loading original data for prediction")
                extractor = DataExtractor()
                df = extractor.load_original_data()
            
            customer_ids = df[FEATURE_CONFIG['customer_id_column']] if FEATURE_CONFIG['customer_id_column'] in df.columns else None
            X, feature_cols = self.preprocess_new_data(df)
            y_pred, y_pred_proba = self.predict_churn(X)
            prediction_df = self.create_prediction_dataframe(
                df, y_pred, y_pred_proba, customer_ids
            )
            summary = self.generate_prediction_summary(prediction_df)
            
            if save_predictions:
                self.save_predictions(prediction_df)
            
            print("=== PREDICTION SUMMARY ===")
            print(f"Total customers: {summary['total_customers']:,}")
            print(f"Predicted churn: {summary['predicted_churn']:,}")
            print(f"Churn rate: {summary['churn_rate']:.2%}")
            print(f"Average churn probability: {summary['avg_churn_probability']:.3f}")
            
            if 'high_risk_customers' in summary:
                print(f"High-risk customers: {summary['high_risk_customers']:,}")
                print(f"High-risk avg probability: {summary['high_risk_avg_probability']:.3f}")
            
            return prediction_df
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise


def main():
    try:
        predictor = ChurnPredictor()
        prediction_df = predictor.predict_batch(save_predictions=True)
        high_risk = predictor.get_high_risk_customers(prediction_df, threshold=0.7)
        
        if not high_risk.empty:
            print(f"\n=== HIGH-RISK CUSTOMERS (Top 10) ===")
            print(high_risk[['CustomerID', 'Churn_Probability', 'Risk_Category']].head(10))
        
        print(f"\nPredictions completed successfully!")
        print(f"Results saved to: {PREDICTIONS_FILE}")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
