import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from pathlib import Path
import psycopg2
from sqlalchemy import create_engine, text
import os
import json
import urllib
from config import FEATURE_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataExtractor:
    def __init__(self):
        pass
 
    def load_from_database(self):
        try:
            # print(os.getcwd())
            with open('src/credentials.json') as f: 
                login = json.load(f)

            username = login['user']
            password = urllib.parse.quote(login['password']) 
            host = login['host']
            port = login['port']
            db_name = login['name']

            conn = create_engine(
            f"postgresql://{username}:{password}@{host}:{port}/{db_name}"
            )

            df = pd.read_sql_query(text("SELECT * FROM customers"), con=conn)
            logger.info("Data loaded from database")
            return df
        except Exception as e:
            logger.error(f"Error loading from database: {str(e)}")
            raise
    
    def get_data_summary(self, df):
        """Return basic summary stats for a dataframe"""
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.astype(str).to_dict()
        }
        return summary
    
    def validate_data(self, df):
        """Check that all required columns exist in the dataframe"""
        required_columns = (
            FEATURE_CONFIG['numerical_features'] + 
            FEATURE_CONFIG['categorical_features'] + 
            [FEATURE_CONFIG['target_column'], FEATURE_CONFIG['customer_id_column']]
        )
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        logger.info("Data validation passed")
        return True


def main():
    extractor = DataExtractor()
    try:
        original_df = extractor.load_from_database()
        is_valid = extractor.validate_data(original_df)
        summary = extractor.get_data_summary(original_df)
        
        print("=== DATA EXTRACTION SUMMARY ===")
        print(summary)
        print("Validation checked passed? --", is_valid)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
