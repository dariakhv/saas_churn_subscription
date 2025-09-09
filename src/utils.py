import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def load_json_config(filepath: Path) -> Dict[str, Any]:
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {filepath}")
        return config
    except Exception as e:
        logger.error(f"Error loading config from {filepath}: {str(e)}")
        raise


def save_json_config(config: Dict[str, Any], filepath: Path) -> None:
    try:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        logger.info(f"Configuration saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving config to {filepath}: {str(e)}")
        raise


def create_feature_summary(df: pd.DataFrame, 
                          numerical_cols: List[str],
                          categorical_cols: List[str]) -> Dict[str, Any]:
    summary = {
        'dataset_info': {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'dtypes': df.dtypes.to_dict()
        },
        'missing_values': df.isnull().sum().to_dict(),
        'numerical_features': {},
        'categorical_features': {}
    }
    
    for col in numerical_cols:
        if col in df.columns:
            summary['numerical_features'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'missing': int(df[col].isnull().sum())
            }
    
    for col in categorical_cols:
        if col in df.columns:
            value_counts = df[col].value_counts()
            summary['categorical_features'][col] = {
                'unique_values': int(value_counts.nunique()),
                'top_values': value_counts.head(5).to_dict(),
                'missing': int(df[col].isnull().sum())
            }
    
    return summary


def plot_feature_distributions(df: pd.DataFrame, 
                             numerical_cols: List[str],
                             categorical_cols: List[str],
                             save_path: Optional[Path] = None) -> None:
    plt.style.use('default')
    sns.set_palette("husl")
    
    if numerical_cols:
        n_numerical = len(numerical_cols)
        cols = min(3, n_numerical)
        rows = (n_numerical + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            if col in df.columns:
                axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        for i in range(n_numerical, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path / 'numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    if categorical_cols:
        n_categorical = len(categorical_cols)
        cols = min(2, n_categorical)
        rows = (n_categorical + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(categorical_cols):
            if col in df.columns:
                value_counts = df[col].value_counts().head(10)
                axes[i].bar(range(len(value_counts)), value_counts.values)
                axes[i].set_title(f'Top 10 values in {col}')
                axes[i].set_xlabel('Values')
                axes[i].set_ylabel('Count')
                axes[i].set_xticks(range(len(value_counts)))
                axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
        
        for i in range(n_categorical, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path / 'categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()


def calculate_correlation_matrix(df: pd.DataFrame, 
                               numerical_cols: List[str],
                               save_path: Optional[Path] = None) -> pd.DataFrame:
    if not numerical_cols:
        logger.warning("No numerical columns provided for correlation analysis")
        return pd.DataFrame()
    
    available_cols = [col for col in numerical_cols if col in df.columns]
    
    if len(available_cols) < 2:
        logger.warning("Need at least 2 numerical columns for correlation analysis")
        return pd.DataFrame()
    
    corr_matrix = df[available_cols].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr_matrix


def generate_data_quality_report(df: pd.DataFrame, 
                               save_path: Optional[Path] = None) -> Dict[str, Any]:
    report = {
        'timestamp': datetime.now().isoformat(),
        'dataset_overview': {
            'shape': df.shape,
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'duplicate_rows': int(df.duplicated().sum()),
            'duplicate_percentage': round(df.duplicated().sum() / len(df) * 100, 2)
        },
        'missing_data_analysis': {
            'total_missing': int(df.isnull().sum().sum()),
            'missing_percentage': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
            'columns_with_missing': df.isnull().sum()[df.isnull().sum() > 0].to_dict()
        },
        'data_types': df.dtypes.value_counts().to_dict(),
        'unique_values_per_column': df.nunique().to_dict()
    }
    
    if save_path:
        report_file = save_path / 'data_quality_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Data quality report saved to {report_file}")
    
    return report


def validate_data_schema(df: pd.DataFrame, 
                        expected_schema: Dict[str, str]) -> Dict[str, Any]:
    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'schema_validation': {},
        'overall_status': 'PASSED'
    }
    
    for column, expected_type in expected_schema.items():
        if column not in df.columns:
            validation_results['schema_validation'][column] = {
                'status': 'FAILED',
                'error': 'Column missing'
            }
            validation_results['overall_status'] = 'FAILED'
        else:
            actual_type = str(df[column].dtype)
            if expected_type.lower() in actual_type.lower():
                validation_results['schema_validation'][column] = {
                    'status': 'PASSED',
                    'expected_type': expected_type,
                    'actual_type': actual_type
                }
            else:
                validation_results['schema_validation'][column] = {
                    'status': 'FAILED',
                    'expected_type': expected_type,
                    'actual_type': actual_type,
                    'error': 'Type mismatch'
                }
                validation_results['overall_status'] = 'FAILED'
    
    return validation_results


def main():
    try:
        from .extract import DataExtractor
        
        extractor = DataExtractor()
        df = extractor.load_original_data()
        
        numerical_cols = ['Age', 'Tenure', 'Usage_Frequency', 'Support_Calls', 'Payment_Delay', 'Total_Spend', 'Last_Interaction']
        categorical_cols = ['Gender', 'Subscription_Type', 'Contract_Length']
        
        summary = create_feature_summary(df, numerical_cols, categorical_cols)
        print("=== FEATURE SUMMARY ===")
        print(json.dumps(summary, indent=2))
        
        quality_report = generate_data_quality_report(df)
        print("\n=== DATA QUALITY REPORT ===")
        print(f"Overall status: {quality_report['overall_status']}")
        print(f"Missing data: {quality_report['missing_data_analysis']['missing_percentage']}%")
        print(f"Duplicate rows: {quality_report['dataset_overview']['duplicate_percentage']}%")
        
        expected_schema = {
            'Age': 'int64',
            'Gender': 'object',
            'Tenure': 'int64',
            'Churn': 'int64'
        }
        schema_validation = validate_data_schema(df, expected_schema)
        print(f"\n=== SCHEMA VALIDATION ===")
        print(f"Status: {schema_validation['overall_status']}")
        
        print("\nUtility functions tested successfully!")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
