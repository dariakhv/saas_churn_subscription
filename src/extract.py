import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from pathlib import Path

try:
    import psycopg2
    from sqlalchemy import create_engine
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logging.warning("Database modules not available. Install with: pip install psycopg2-binary sqlalchemy")

from .config import (
    ORIGINAL_DATA, SYNTHETIC_SUBSCRIPTIONS, SYNTHETIC_INVOICES,
    DATABASE_CONFIG, FEATURE_CONFIG
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExtractor:
    def __init__(self):
        self.original_data = None
        self.subscriptions = None
        self.invoices = None
    
    def load_original_data(self) -> pd.DataFrame:
        try:
            logger.info(f"Loading original data from {ORIGINAL_DATA}")
            df = pd.read_csv(ORIGINAL_DATA)
            df.columns = df.columns.str.replace(' ', '_').str.replace('\n', '')
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            self.original_data = df
            return df
        except FileNotFoundError:
            logger.error(f"Original data file not found: {ORIGINAL_DATA}")
            raise
        except Exception as e:
            logger.error(f"Error loading original data: {str(e)}")
            raise
    
    def load_synthetic_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            logger.info("Loading synthetic data...")
            if SYNTHETIC_SUBSCRIPTIONS.exists():
                self.subscriptions = pd.read_csv(SYNTHETIC_SUBSCRIPTIONS)
                logger.info(f"Loaded {len(self.subscriptions)} subscription records")
            else:
                logger.warning("Synthetic subscriptions file not found")
                self.subscriptions = pd.DataFrame()
            
            if SYNTHETIC_INVOICES.exists():
                self.invoices = pd.read_csv(SYNTHETIC_INVOICES)
                logger.info(f"Loaded {len(self.invoices)} invoice records")
            else:
                logger.warning("Synthetic invoices file not found")
                self.invoices = pd.DataFrame()
            
            return self.subscriptions, self.invoices
        except Exception as e:
            logger.error(f"Error loading synthetic data: {str(e)}")
            raise
    
    def load_from_database(self, query: str) -> pd.DataFrame:
        if not DB_AVAILABLE:
            raise ImportError("Database modules not available")
        try:
            logger.info("Connecting to database...")
            conn_string = (
                f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
                f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}"
                f"/{DATABASE_CONFIG['database']}"
            )
            engine = create_engine(conn_string)
            df = pd.read_sql(query, engine)
            logger.info(f"Loaded {len(df)} records from database")
            return df
        except Exception as e:
            logger.error(f"Error loading from database: {str(e)}")
            raise
    
    def get_data_summary(self) -> Dict:
        summary = {}
        if self.original_data is not None:
            summary['original_data'] = {
                'shape': self.original_data.shape,
                'columns': list(self.original_data.columns),
                'missing_values': self.original_data.isnull().sum().to_dict(),
                'dtypes': self.original_data.dtypes.to_dict()
            }
        if self.subscriptions is not None and len(self.subscriptions) > 0:
            summary['subscriptions'] = {
                'shape': self.subscriptions.shape,
                'columns': list(self.subscriptions.columns),
                'missing_values': self.subscriptions.isnull().sum().to_dict()
            }
        if self.invoices is not None and len(self.invoices) > 0:
            summary['invoices'] = {
                'shape': self.invoices.shape,
                'columns': list(self.invoices.columns),
                'missing_values': self.invoices.isnull().sum().to_dict()
            }
        return summary
    
    def validate_data(self) -> bool:
        if self.original_data is None:
            logger.error("Original data not loaded")
            return False
        required_columns = (
            FEATURE_CONFIG['numerical_features'] + 
            FEATURE_CONFIG['categorical_features'] + 
            [FEATURE_CONFIG['target_column'], FEATURE_CONFIG['customer_id_column']]
        )
        missing_columns = [col for col in required_columns if col not in self.original_data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        logger.info("Data validation passed")
        return True


def main():
    extractor = DataExtractor()
    try:
        original_df = extractor.load_original_data()
        subscriptions_df, invoices_df = extractor.load_synthetic_data()
        is_valid = extractor.validate_data()
        summary = extractor.get_data_summary()
        
        print("=== DATA EXTRACTION SUMMARY ===")
        for dataset, info in summary.items():
            print(f"\n{dataset.upper()}:")
            print(f"  Shape: {info['shape']}")
            print(f"  Columns: {len(info['columns'])}")
            if 'missing_values' in info:
                missing = {k: v for k, v in info['missing_values'].items() if v > 0}
                if missing:
                    print(f"  Missing values: {missing}")
        print(f"\nData validation: {'PASSED' if is_valid else 'FAILED'}")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
