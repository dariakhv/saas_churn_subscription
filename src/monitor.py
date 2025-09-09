import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import pickle

try:
    from evidently import Report
    from evidently.metrics import (
        ValueDrift, DriftedColumnsCount, DatasetMissingValueCount,
        RowCount, ColumnCount, MissingValueCount
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logging.warning("Evidently not available. Install with: pip install evidently")

from .config import DRIFT_CONFIG, DRIFT_REPORT_FILE, METRICS_FILE, FEATURE_CONFIG
from .extract import DataExtractor
from .preprocess import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor:
    def __init__(self):
        if not EVIDENTLY_AVAILABLE:
            raise ImportError("Evidently is required for model monitoring")
        self.reference_data = None
        self.current_data = None
        self.drift_results = {}
        
    def load_reference_data(self, data_path: Optional[Path] = None) -> pd.DataFrame:
        try:
            if data_path:
                logger.info(f"Loading reference data from {data_path}")
                self.reference_data = pd.read_csv(data_path)
            else:
                logger.info("Loading original data as reference")
                extractor = DataExtractor()
                df = extractor.load_original_data()
                preprocessor = DataPreprocessor()
                df_clean = preprocessor.clean_data(df)
                df_encoded = preprocessor.encode_categorical_features(df_clean)
                df_scaled = preprocessor.scale_numerical_features(df_encoded)
                df_features = preprocessor.create_derived_features(df_scaled)
                
                if len(df_features) > DRIFT_CONFIG['reference_dataset_size']:
                    self.reference_data = df_features.sample(
                        n=DRIFT_CONFIG['reference_dataset_size'],
                        random_state=42
                    )
                else:
                    self.reference_data = df_features
            
            logger.info(f"Reference data loaded: {self.reference_data.shape}")
            return self.reference_data
        except Exception as e:
            logger.error(f"Error loading reference data: {str(e)}")
            raise
    
    def load_current_data(self, data_path: Path) -> pd.DataFrame:
        try:
            logger.info(f"Loading current data from {data_path}")
            df = pd.read_csv(data_path)
            
            preprocessor_path = Path("models/preprocessor.pkl")
            if preprocessor_path.exists():
                with open(preprocessor_path, 'rb') as f:
                    preprocessor = pickle.load(f)
                logger.info("Loaded fitted preprocessor for current data")
            else:
                logger.warning("Fitted preprocessor not found, using new preprocessor")
                preprocessor = DataPreprocessor()
            
            df_clean = preprocessor.clean_data(df)
            df_encoded = preprocessor.encode_categorical_features(df_clean)
            df_scaled = preprocessor.scale_numerical_features(df_encoded, fit=False)
            df_features = preprocessor.create_derived_features(df_scaled)
            
            self.current_data = df_features
            logger.info(f"Current data loaded: {self.current_data.shape}")
            return self.current_data
        except Exception as e:
            logger.error(f"Error loading current data: {str(e)}")
            raise
    
    def detect_data_drift(self) -> Dict[str, any]:
        if self.reference_data is None or self.current_data is None:
            logger.error("Reference and current data must be loaded first")
            return {}
        
        logger.info("Detecting data drift...")
        drift_report = Report(metrics=[
            ValueDrift(column=col) for col in DRIFT_CONFIG['monitoring_columns'] 
            if col in self.reference_data.columns
        ])
        
        drift_report.run(
            reference_data=self.reference_data,
            current_data=self.current_data
        )
        
        drift_results = {}
        for metric in drift_report.metrics:
            if hasattr(metric, 'result'):
                result = metric.result
                drift_results[metric.column_name] = {
                    'drift_detected': getattr(result, 'drift_detected', False),
                    'drift_score': getattr(result, 'drift_score', 0.0),
                    'details': str(result)
                }
        
        self.drift_results = drift_results
        logger.info(f"Data drift detection completed. Found {len(drift_results)} drifted columns")
        return drift_results
    
    def detect_target_drift(self) -> Dict:
        if self.reference_data is None or self.current_data is None:
            logger.error("Reference and current data must be loaded first")
            return {}
        
        target_col = FEATURE_CONFIG['target_column']
        if target_col not in self.reference_data.columns:
            logger.warning("Target column not available for drift detection")
            return {}
        
        logger.info("Detecting target drift...")
        target_drift_report = Report(metrics=[
            ValueDrift(column=target_col)
        ])
        
        target_drift_report.run(
            reference_data=self.reference_data,
            current_data=self.current_data
        )
        
        target_metrics = target_drift_report.metrics
        target_drift_results = {}
        
        for metric in target_metrics:
            if hasattr(metric, 'result'):
                result = metric.result
                target_drift_results[metric.column_name] = {
                    'drift_detected': getattr(result, 'drift_detected', False),
                    'drift_score': getattr(result, 'drift_score', 0.0),
                    'details': str(result)
                }
        
        logger.info("Target drift detection completed")
        return target_drift_results
    
    def generate_drift_report(self, save_html: bool = True) -> str:
        if not self.drift_results:
            logger.warning("No drift results available. Run detect_data_drift first.")
            return ""
        
        logger.info("Generating drift report...")
        metrics = []
        for col in DRIFT_CONFIG['monitoring_columns']:
            if col in self.reference_data.columns:
                metrics.append(ValueDrift(column=col))
        
        target_col = FEATURE_CONFIG['target_column']
        if target_col in self.reference_data.columns:
            metrics.append(ValueDrift(column=target_col))
        
        report = Report(metrics=metrics)
        report.run(
            reference_data=self.reference_data,
            current_data=self.current_data
        )
        
        if save_html:
            report.save_html(DRIFT_REPORT_FILE)
            logger.info(f"Drift report saved to {DRIFT_REPORT_FILE}")
        
        return str(report)
    
    def check_drift_thresholds(self) -> Dict[str, bool]:
        if not self.drift_results:
            logger.warning("No drift results available")
            return {}
        
        threshold = DRIFT_CONFIG['drift_threshold']
        alerts = {}
        
        for metric_name, result in self.drift_results.items():
            if result.get('drift_detected', False):
                drift_score = result.get('drift_score', 0)
                if drift_score > threshold:
                    alerts[metric_name] = True
                    logger.warning(f"High drift detected in {metric_name}: {drift_score:.3f}")
                else:
                    alerts[metric_name] = False
            else:
                alerts[metric_name] = False
        
        return alerts
    
    def save_monitoring_results(self, filepath: Optional[Path] = None) -> None:
        filepath = filepath or METRICS_FILE
        monitoring_results = {
            'timestamp': datetime.now().isoformat(),
            'drift_results': self.drift_results,
            'data_info': {
                'reference_shape': self.reference_data.shape if self.reference_data is not None else None,
                'current_shape': self.current_data.shape if self.current_data is not None else None
            },
            'drift_threshold': DRIFT_CONFIG['drift_threshold']
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(monitoring_results, f, indent=2, default=str)
            logger.info(f"Monitoring results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving monitoring results: {str(e)}")
            raise


def main():
    try:
        monitor = ModelMonitor()
        reference_data = monitor.load_reference_data()
        current_data = monitor.load_current_data(
            Path("data/customers_clean.csv")
        )
        drift_results = monitor.detect_data_drift()
        target_drift = monitor.detect_target_drift()
        alerts = monitor.check_drift_thresholds()
        report = monitor.generate_drift_report()
        monitor.save_monitoring_results()
        
        print("=== MONITORING COMPLETED ===")
        print(f"Drift metrics found: {len(drift_results)}")
        print(f"Target drift metrics: {len(target_drift)}")
        print(f"High drift alerts: {sum(alerts.values())}")
        print(f"Drift report saved to: {DRIFT_REPORT_FILE}")
        print(f"Results saved to: {METRICS_FILE}")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
