#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import pickle

sys.path.append(str(Path(__file__).parent / "src"))
from src.config import OUTPUTS_DIR, PREDICTIONS_FILE, MODEL_FILE, FEATURE_CONFIG
from src.extract import DataExtractor
from src.preprocess import DataPreprocessor
from src.train import ModelTrainer
from src.monitor import ModelMonitor
from src.utils import setup_logging


def setup_pipeline_logging(log_level: str = "INFO") -> None:
    log_file = OUTPUTS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_level, log_file)
    logging.info("Pipeline logging initialized")


def run_data_extraction() -> pd.DataFrame:
    logging.info("=== STEP 1: DATA EXTRACTION ===")
    try:
        extractor = DataExtractor()
        df = extractor.load_from_database()
        is_valid = extractor.validate_data(df)
        if not is_valid:
            raise ValueError("Data validation failed")
        summary = extractor.get_data_summary(df)
        logging.info("Data extraction completed successfully")
        return df, summary
    except Exception as e:
        logging.error(f"Data extraction failed: {str(e)}")
        raise


def run_data_preprocessing(df: pd.DataFrame) -> tuple:
    logging.info("=== STEP 2: DATA PREPROCESSING ===")
    try:
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(df)
        X = df_clean.drop(columns=FEATURE_CONFIG['target_column'])
        y = df_clean[FEATURE_CONFIG['target_column']]
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        preprocessor.encode_features(X_train)
        logging.info("Data preprocessing completed successfully")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Data preprocessing failed: {str(e)}")
        raise


def run_model_training(X_train, X_test, y_train, y_test) -> dict:
    logging.info("=== STEP 3: MODEL TRAINING / SELECTION ===")
    try:
        trainer = ModelTrainer()
        optim_model_path = Path("models/optim_model.pkl")
        rf_model_path = Path("models/RF_best_model.pkl")
        pipe_path = Path("models/pipe.pkl")
        if optim_model_path.exists():
            with open(optim_model_path, "rb") as f:
                model = pickle.load(f)
            metrics = trainer.evaluate_model(model, X_test, y_test)
        elif rf_model_path.exists():
            with open(rf_model_path, "rb") as f:
                model = pickle.load(f)
            trainer.train_model(model, X_train, y_train)
            metrics = trainer.evaluate_model(model, X_test, y_test)
        elif pipe_path.exists():
            with open(pipe_path, "rb") as f:
                pipe = pickle.load(f)
            model = trainer.hyperparameter_optimization(pipe, X_train, y_train)
            trainer.train_model(model, X_train, y_train)
            metrics = trainer.evaluate_model(model, X_test, y_test)
        else:
            raise FileNotFoundError("No saved models or pipeline found in 'models/'.")
        logging.info("Model training/selection completed")
        return metrics
    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")
        raise


def run_model_monitoring() -> dict:
    logging.info("=== STEP 4: MODEL MONITORING ===")
    try:
        monitor = ModelMonitor()
        monitor.load_reference_data()
        monitor.load_current_data()
        drift_results = monitor.detect_data_drift()
        target_drift = monitor.detect_target_drift()
        alerts = monitor.check_drift_thresholds()
    
        logging.info("Model monitoring completed successfully")
        return {"drift_results": drift_results, "target_drift": target_drift, "alerts": alerts}
    except Exception as e:
        logging.warning(f"Model monitoring failed: {str(e)}")
        return {"error": str(e)}


def generate_pipeline_report(extraction_summary: dict, training_metrics: dict, monitoring_results: dict) -> None:
    logging.info("=== GENERATING PIPELINE REPORT ===")
    try:
        report = {
            'pipeline_execution': {
                'timestamp': datetime.now().isoformat(),
                'status': 'COMPLETED',
                'steps_completed': [
                    'Data Extraction',
                    'Data Preprocessing', 
                    'Model Training/Selection',
                    'Model Monitoring'
                ]
            },
            'data_summary': extraction_summary,
            'model_performance': training_metrics,
            'monitoring_results': monitoring_results
        }
        report_file = OUTPUTS_DIR / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logging.info(f"Pipeline report saved to {report_file}")
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Model saved to: {MODEL_FILE}")
        print(f"Pipeline report: {report_file}")
        print("="*60)
    except Exception as e:
        logging.error(f"Pipeline report generation failed: {str(e)}")
        raise




def main():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        setup_pipeline_logging(args.log_level)
        logging.info("Starting Customer Churn Prediction Pipeline")
        
        # Step 1: Data Extraction
        df, extraction_summary = run_data_extraction()
        # Step 2: Data Preprocessing
        X_train, X_test, y_train, y_test = run_data_preprocessing(df)
        # Step 3: Model Training/Selection and Evaluation
        training_metrics = run_model_training(X_train, X_test, y_train, y_test)
        # Step 4: Model Monitoring
        monitoring_results = run_model_monitoring()
        # Final report
        generate_pipeline_report(extraction_summary, training_metrics, monitoring_results)
        
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        print(f"\n PIPELINE FAILED: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
