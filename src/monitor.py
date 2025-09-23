import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime
import pickle
from evidently import Report
from evidently.metrics import (
    ValueDrift, DriftedColumnsCount, DatasetMissingValueCount,
    RowCount, ColumnCount, MissingValueCount
)
from config import DRIFT_CONFIG, METRICS_FILE, FEATURE_CONFIG
from extract import DataExtractor
from preprocess import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor:
    def __init__(self):
        self.reference_data = None
        self.current_data = None
        self.drift_results = {}
        
    def load_reference_data(self) -> pd.DataFrame:
        try:
            logger.info("Loading reference data from database")
            extractor = DataExtractor()
            df = extractor.load_from_database()
            preprocessor = DataPreprocessor()
            df_clean = preprocessor.clean_data(df)
            cols = [c for c in DRIFT_CONFIG['monitoring_columns'] if c in df_clean.columns]
            self.reference_data = df_clean[cols].copy() if cols else df_clean.copy()
            logger.info(f"Reference data loaded: {self.reference_data.shape}")
            return self.reference_data
        except Exception as e:
            logger.error(f"Error loading reference data: {str(e)}")
            raise
    
    def load_current_data(self) -> pd.DataFrame:
        try:
            logger.info("Loading current data from database")
            extractor = DataExtractor()
            df = extractor.load_from_database()
            preprocessor = DataPreprocessor()
            df_clean = preprocessor.clean_data(df)
            cols = [c for c in DRIFT_CONFIG['monitoring_columns'] if c in df_clean.columns]
            self.current_data = df_clean[cols].copy() if cols else df_clean.copy()
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
        if self.reference_data is None or self.current_data is None:
            logger.warning("No data available. Load reference and current data first.")
            return ""
        logger.info("Generating drift report...")
        metrics = []
        for col in DRIFT_CONFIG['monitoring_columns']:
            if col in self.reference_data.columns:
                metrics.append(ValueDrift(column=col))
        target_col = FEATURE_CONFIG['target_column']
        if target_col in self.reference_data.columns and target_col in self.current_data.columns:
            metrics.append(ValueDrift(column=target_col))
        report = Report(metrics=metrics)
        report.run(reference_data=self.reference_data, current_data=self.current_data)
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

def main():
    try:
        monitor = ModelMonitor()
        monitor.load_reference_data()
        monitor.load_current_data()
        drift_results = monitor.detect_data_drift()
        target_drift = monitor.detect_target_drift()
        alerts = monitor.check_drift_thresholds()
        print("=== MONITORING COMPLETED ===")
        print(f"Drift metrics found: {len(drift_results)}")
        print(f"Target drift metrics: {len(target_drift)}")
        print(f"High drift alerts: {sum(alerts.values())}")
 
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
