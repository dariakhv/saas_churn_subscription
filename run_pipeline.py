#!/usr/bin/env python3
"""
Main pipeline runner for the Customer Churn Prediction Pipeline

This script orchestrates the entire ML pipeline:
1. Data extraction and validation
2. Data preprocessing and feature engineering
3. Model training and evaluation
4. Batch predictions
5. Model monitoring and drift detection
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import *
from src.extract import DataExtractor
from src.preprocess import DataPreprocessor
from src.train import ModelTrainer
from src.predict import ChurnPredictor
from src.monitor import ModelMonitor
from src.utils import setup_logging, generate_data_quality_report


def setup_pipeline_logging(log_level: str = "INFO") -> None:
    """Setup logging for the pipeline"""
    log_file = OUTPUTS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_level, log_file)
    logging.info("Pipeline logging initialized")


def run_data_extraction() -> tuple:
    """Run data extraction step"""
    logging.info("=== STEP 1: DATA EXTRACTION ===")
    
    try:
        extractor = DataExtractor()
        
        # Load original data
        original_df = extractor.load_original_data()
        
        # Load synthetic data
        subscriptions_df, invoices_df = extractor.load_synthetic_data()
        
        # Validate data
        is_valid = extractor.validate_data()
        if not is_valid:
            raise ValueError("Data validation failed")
        
        # Get data summary
        summary = extractor.get_data_summary()
        logging.info("Data extraction completed successfully")
        
        return original_df, subscriptions_df, invoices_df, summary
        
    except Exception as e:
        logging.error(f"Data extraction failed: {str(e)}")
        raise


def run_data_preprocessing(original_df: pd.DataFrame) -> tuple:
    """Run data preprocessing step"""
    logging.info("=== STEP 2: DATA PREPROCESSING ===")
    
    try:
        preprocessor = DataPreprocessor()
        
        # Clean data
        df_clean = preprocessor.clean_data(original_df)
        
        # Encode categorical features
        df_encoded = preprocessor.encode_categorical_features(df_clean)
        
        # Scale numerical features
        df_scaled = preprocessor.scale_numerical_features(df_encoded)
        
        # Create derived features
        df_features = preprocessor.create_derived_features(df_scaled)
        
        # Prepare features for modeling
        X, feature_cols = preprocessor.prepare_features(df_features)
        y = df_features[FEATURE_CONFIG['target_column']]
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        logging.info("Data preprocessing completed successfully")
        
        return (X_train, X_val, X_test, y_train, y_val, y_test, 
                feature_cols, preprocessor)
        
    except Exception as e:
        logging.error(f"Data preprocessing failed: {str(e)}")
        raise


def run_model_training(X_train: pd.DataFrame, X_val: pd.DataFrame, 
                      X_test: pd.DataFrame, y_train: pd.Series, 
                      y_val: pd.Series, y_test: pd.Series) -> tuple:
    """Run model training step"""
    logging.info("=== STEP 3: MODEL TRAINING ===")
    
    try:
        trainer = ModelTrainer()
        
        # Train all models
        metrics = trainer.train_all_models(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Save best model
        trainer.save_model(trainer.best_model, MODEL_FILE)
        
        # Save metrics
        trainer.save_metrics(METRICS_FILE)
        
        logging.info("Model training completed successfully")
        
        return trainer.best_model, trainer.best_model_name, metrics
        
    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")
        raise


def run_predictions(best_model_name: str) -> pd.DataFrame:
    """Run batch predictions step"""
    logging.info("=== STEP 4: BATCH PREDICTIONS ===")
    
    try:
        predictor = ChurnPredictor()
        
        # Make batch predictions
        prediction_df = predictor.predict_batch(save_predictions=True)
        
        # Get high-risk customers
        high_risk = predictor.get_high_risk_customers(prediction_df, threshold=0.7)
        
        logging.info("Batch predictions completed successfully")
        
        return prediction_df, high_risk
        
    except Exception as e:
        logging.error(f"Batch predictions failed: {str(e)}")
        raise


def run_model_monitoring() -> dict:
    """Run model monitoring step"""
    logging.info("=== STEP 5: MODEL MONITORING ===")
    
    try:
        monitor = ModelMonitor()
        
        # Load reference data
        reference_data = monitor.load_reference_data()
        
        # Load current data (using original data as example)
        current_data = monitor.load_current_data(ORIGINAL_DATA)
        
        # Detect data drift
        drift_results = monitor.detect_data_drift()
        
        # Detect target drift
        target_drift = monitor.detect_target_drift()
        
        # Check drift thresholds
        alerts = monitor.check_drift_thresholds()
        
        # Generate drift report
        report = monitor.generate_drift_report()
        
        # Save monitoring results
        monitor.save_monitoring_results()
        
        logging.info("Model monitoring completed successfully")
        
        return {
            'drift_results': drift_results,
            'target_drift': target_dift,
            'alerts': alerts
        }
        
    except Exception as e:
        logging.error(f"Model monitoring failed: {str(e)}")
        raise


def generate_pipeline_report(extraction_summary: dict, training_metrics: dict, 
                           prediction_summary: dict, monitoring_results: dict) -> None:
    """Generate comprehensive pipeline report"""
    logging.info("=== GENERATING PIPELINE REPORT ===")
    
    try:
        report = {
            'pipeline_execution': {
                'timestamp': datetime.now().isoformat(),
                'status': 'COMPLETED',
                'steps_completed': [
                    'Data Extraction',
                    'Data Preprocessing', 
                    'Model Training',
                    'Batch Predictions',
                    'Model Monitoring'
                ]
            },
            'data_summary': extraction_summary,
            'model_performance': training_metrics,
            'predictions_summary': prediction_summary,
            'monitoring_results': monitoring_results
        }
        
        # Save pipeline report
        report_file = OUTPUTS_DIR / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logging.info(f"Pipeline report saved to {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best Model: {training_metrics.get('best_model', 'Unknown')}")
        print(f"Test F1-Score: {training_metrics.get('test_metrics', {}).get('f1_score', 0):.4f}")
        print(f"Test Accuracy: {training_metrics.get('test_metrics', {}).get('accuracy', 0):.4f}")
        print(f"High Drift Alerts: {sum(monitoring_results.get('alerts', {}).values())}")
        print(f"Predictions saved to: {PREDICTIONS_FILE}")
        print(f"Model saved to: {MODEL_FILE}")
        print(f"Drift report: {DRIFT_REPORT_FILE}")
        print(f"Pipeline report: {report_file}")
        print("="*60)
        
    except Exception as e:
        logging.error(f"Pipeline report generation failed: {str(e)}")
        raise


def main():
    """Main pipeline execution function"""
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--skip-monitoring", action="store_true",
                       help="Skip model monitoring step")
    parser.add_argument("--data-path", type=str,
                       help="Custom path to input data")
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        setup_pipeline_logging(args.log_level)
        logging.info("Starting Customer Churn Prediction Pipeline")
        
        # Step 1: Data Extraction
        original_df, subscriptions_df, invoices_df, extraction_summary = run_data_extraction()
        
        # Step 2: Data Preprocessing
        (X_train, X_val, X_test, y_train, y_val, y_test, 
         feature_cols, preprocessor) = run_data_preprocessing(original_df)
        
        # Step 3: Model Training
        best_model, best_model_name, training_metrics = run_model_training(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Step 4: Batch Predictions
        prediction_df, high_risk = run_predictions(best_model_name)
        prediction_summary = {
            'total_customers': len(prediction_df),
            'predicted_churn': int(prediction_df['Churn_Prediction'].sum()),
            'churn_rate': float(prediction_df['Churn_Prediction'].mean()),
            'high_risk_customers': len(high_risk)
        }
        
        # Step 5: Model Monitoring (optional)
        monitoring_results = {}
        if not args.skip_monitoring:
            try:
                monitoring_results = run_model_monitoring()
            except Exception as e:
                logging.warning(f"Model monitoring failed, continuing: {str(e)}")
                monitoring_results = {'error': str(e)}
        else:
            logging.info("Model monitoring skipped")
            monitoring_results = {'skipped': True}
        
        # Generate final report
        generate_pipeline_report(
            extraction_summary, training_metrics, 
            prediction_summary, monitoring_results
        )
        
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        print(f"\n❌ PIPELINE FAILED: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
