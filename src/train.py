import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, Tuple, Any
from pathlib import Path
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_fscore_support, accuracy_score
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

from .config import MODEL_PARAMS, MODEL_CONFIG, MODEL_FILE, METRICS_FILE, FEATURE_CONFIG
from .extract import DataExtractor
from .preprocess import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.metrics = {}
        
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        logger.info("Training Random Forest model...")
        rf = RandomForestClassifier(**MODEL_PARAMS['random_forest'])
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        logger.info("Random Forest training completed")
        return rf
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series):
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, skipping XGBoost training")
            return None
        logger.info("Training XGBoost model...")
        xgb_model = xgb.XGBClassifier(**MODEL_PARAMS['xgboost'])
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        logger.info("XGBoost training completed")
        return xgb_model
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        logger.info("Training Logistic Regression model...")
        lr = LogisticRegression(**MODEL_PARAMS['logistic_regression'])
        lr.fit(X_train, y_train)
        self.models['logistic_regression'] = lr
        logger.info("Logistic Regression training completed")
        return lr
    
    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                      dataset_name: str) -> Dict[str, float]:
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_recall_fscore_support(y, y_pred, average='weighted')[0],
            'recall': precision_recall_fscore_support(y, y_pred, average='weighted')[1],
            'f1_score': precision_recall_fscore_support(y, y_pred, average='weighted')[2]
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
        
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        report = classification_report(y, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        logger.info(f"{dataset_name} metrics for {type(model).__name__}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> float:
        cv_scores = cross_val_score(
            model, X, y, 
            cv=MODEL_CONFIG['cv_folds'], 
            scoring='f1_weighted'
        )
        mean_cv_score = cv_scores.mean()
        std_cv_score = cv_scores.std()
        logger.info(f"Cross-validation F1-score: {mean_cv_score:.4f} (+/- {std_cv_score * 2:.4f})")
        return mean_cv_score
    
    def select_best_model(self, X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Any, str]:
        logger.info("Selecting best model based on validation performance...")
        best_score = -1
        best_model = None
        best_model_name = None
        
        for name, model in self.models.items():
            metrics = self.evaluate_model(model, X_val, y_val, "Validation")
            f1_score = metrics['f1_score']
            
            if f1_score > best_score:
                best_score = f1_score
                best_model = model
                best_model_name = name
        
        self.best_model = best_model
        self.best_model_name = best_model_name
        logger.info(f"Best model: {best_model_name} with F1-score: {best_score:.4f}")
        return best_model, best_model_name
    
    def get_feature_importance(self, model: Any, preprocessor=None) -> pd.DataFrame:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            logger.warning("Model doesn't have feature importance or coefficients")
            return pd.DataFrame()
        
        if preprocessor is None or not hasattr(preprocessor, 'feature_names'):
            logger.warning("Preprocessor not available, using generic feature names")
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        else:
            feature_names = preprocessor.feature_names
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        return importance_df
    
    def save_model(self, model: Any, filepath: Path) -> None:
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def save_preprocessor(self, preprocessor: Any, filepath: Path) -> None:
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(preprocessor, f)
            logger.info(f"Preprocessor saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving preprocessor: {str(e)}")
            raise
    
    def save_metrics(self, filepath: Path) -> None:
        try:
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
            logger.info(f"Metrics saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            raise
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        preprocessor=None) -> Dict[str, Any]:
        logger.info("Starting model training pipeline...")
        self.train_random_forest(X_train, y_train)
        if XGBOOST_AVAILABLE:
            self.train_xgboost(X_train, y_train)
        self.train_logistic_regression(X_train, y_train)
        
        cv_scores = {}
        for name, model in self.models.items():
            cv_scores[name] = self.cross_validate_model(model, X_train, y_train)
        
        best_model, best_model_name = self.select_best_model(X_val, y_val)
        test_metrics = self.evaluate_model(best_model, X_test, y_test, "Test")
        feature_importance = self.get_feature_importance(best_model, preprocessor)
        
        self.metrics = {
            'cross_validation_scores': cv_scores,
            'best_model': best_model_name,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance.to_dict('records') if not feature_importance.empty else [],
            'model_config': MODEL_CONFIG,
            'training_info': {
                'train_samples': len(X_train),
                'validation_samples': len(X_val),
                'test_samples': len(X_test),
                'features': len(X_train.columns)
            }
        }
        return self.metrics


def main():
    try:
        extractor = DataExtractor()
        df = extractor.load_original_data()
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(df)
        df_encoded = preprocessor.encode_categorical_features(df_clean)
        df_scaled = preprocessor.scale_numerical_features(df_encoded)
        df_features = preprocessor.create_derived_features(df_scaled)
        X, feature_cols = preprocessor.prepare_features(df_features)
        y = df_features[FEATURE_CONFIG['target_column']][preprocessor.valid_rows]
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        trainer = ModelTrainer()
        metrics = trainer.train_all_models(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor)
        trainer.save_model(trainer.best_model, MODEL_FILE)
        trainer.save_preprocessor(preprocessor, MODEL_FILE.parent / "preprocessor.pkl")
        trainer.save_metrics(METRICS_FILE)
        
        print("=== TRAINING COMPLETED ===")
        print(f"Best model: {trainer.best_model_name}")
        print(f"Test F1-score: {metrics['test_metrics']['f1_score']:.4f}")
        print(f"Test Accuracy: {metrics['test_metrics']['accuracy']:.4f}")
        print(f"Model saved to: {MODEL_FILE}")
        print(f"Metrics saved to: {METRICS_FILE}")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
