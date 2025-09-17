import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

ORIGINAL_DATA = DATA_DIR / "customers_clean.csv"
SYNTHETIC_SUBSCRIPTIONS = DATA_DIR / "subscriptions_clean.csv"
SYNTHETIC_INVOICES = DATA_DIR / "invoices_clean.csv"

MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "validation_size": 0.2,
    "cv_folds": 5
}

FEATURE_CONFIG = {
    "numerical_features": [
        "age", "tenure", "usage_frequency", "support_calls", 
        "payment_delay", "total_spend", "last_interaction"
    ],
    "ordinal_features": [
        "subscription_type", "contract_length"
    ],
    "drop_features": ["gender"],
    "target_column": "churn",
    "customer_id_column": "customerid"
}

MODEL_PARAMS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    },
    "logistic_regression": {
        "C": 1.0,
        "random_state": 42,
        "max_iter": 1000
    }
}

DRIFT_CONFIG = {
    "reference_dataset_size": 10000,
    "drift_threshold": 0.1,
    "monitoring_columns": [
        "age", "tenure", "usage_frequency", "support_calls",
        "payment_delay", "total_spend", "last_interaction"
    ]
}

DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "saas_proj"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "Nhjkkmm899")
}

PREDICTIONS_FILE = OUTPUTS_DIR / "predictions.csv"
DRIFT_REPORT_FILE = OUTPUTS_DIR / "drift_report.html"
METRICS_FILE = OUTPUTS_DIR / "metrics.json"
MODEL_FILE = MODELS_DIR / "churn_model.pkl"

for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    directory.mkdir(exist_ok=True)
