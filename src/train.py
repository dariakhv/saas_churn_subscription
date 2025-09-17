import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, Tuple, Any
from pathlib import Path
import pickle
import os

from scipy.stats import loguniform, randint


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    ConfusionMatrixDisplay, 
    make_scorer, recall_score, f1_score
)
from sklearn.model_selection import RandomizedSearchCV
from config import MODEL_PARAMS, MODEL_CONFIG, MODEL_FILE, METRICS_FILE, FEATURE_CONFIG
from extract import DataExtractor
from preprocess import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        self.metrics = None
        
    def hyperparameter_optimization(self, pipe, X_train, y_train):
        param_grid = {
            "randomforestclassifier__n_estimators": randint(100, 1000),          # number of trees
            "randomforestclassifier__max_depth": randint(3, 10),                 # depth of trees      
            "randomforestclassifier__max_features": ["sqrt", "log2", None],      # number of features at each split               
            "randomforestclassifier__class_weight": [None, "balanced"],          # balance classes or not
            "randomforestclassifier__max_samples": loguniform(0.5, 1.0)          # fraction of dataset if bootstrap=True
        }

        random_search = RandomizedSearchCV(
            pipe,
            param_grid,
            n_iter=10,
            n_jobs=-1,
            random_state=123,
            return_train_score=True,
            scoring=make_scorer(recall_score)
        )
        
        random_search.fit(X_train, y_train)

        print("Best Recall Score:", random_search.best_score_)
        print("Best Estimator:", random_search.best_estimator_)

        # Save the model
        with open("models/RF_best_model.pkl", "wb") as file:
            pickle.dump(random_search.best_estimator_, file)
    
        self.best_model = random_search.best_estiamtor_
        return self.best_model

    def train_model(self, model, X_train, y_train):
        
        model.fit(X_train, y_train)

        # If pipeline, extract feature importances from last step
        if hasattr(self.best_model, "named_steps"):
            clf = self.best_model.named_steps.get("clf")
        else:
            clf = self.best_model

        if hasattr(clf, "feature_importances_"):
            self.feature_importance = clf.feature_importances_
        
        model_path = "models/optim_model.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        pickle.dump(model, open(model_path, "wb"))

        return model

    def evaluate_model(self, model, X_test, y_test):

        y_pred = model.predict(X_test)
        # y_prob = model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        print("Evaluation Metrics:", self.metrics)
        return self.metrics
               
def main():
    try:
        extractor = DataExtractor()
        df = extractor.load_from_database()
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(df)
        X = df_clean.drop(columns = FEATURE_CONFIG['target_column'])
        y = df_clean[FEATURE_CONFIG['target_column']]
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        
        trainer = ModelTrainer()
        print(os.getcwd())
        optim_model_path = "models/optim_model.pkl"
        rf_model_path = "models/RF_best_model.pkl"
        pipe_path = "models/pipe.pkl"

        if os.path.exists(optim_model_path):
            # Load and evaluate optimized model
            with open(optim_model_path, "rb") as f:
                model = pickle.load(f)
            metrics = trainer.evaluate_model(model, X_test, y_test)

        elif os.path.exists(rf_model_path):
            # Load RF best model and retrain
            with open(rf_model_path, "rb") as f:
                model = pickle.load(f)
            trainer.train_model(model, X_train, y_train)
            metrics = trainer.evaluate_model(model, X_test, y_test)

        elif os.path.exists(pipe_path):
            # Load pipeline and optimize
            with open(pipe_path, "rb") as f:
                pipe = pickle.load(f)
            model = trainer.hyperparameter_optimization(pipe, X_train, y_train)
            trainer.train_model(model, X_train, y_train)
            metrics = trainer.evaluate_model(model, X_test, y_test)

        else:
            raise FileNotFoundError("No saved models or pipeline found in 'models/'.")

        print("Final metrics:", metrics)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
