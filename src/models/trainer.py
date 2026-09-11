"""Model training pipeline with 5 classification models and automated MLflow tracking."""
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
import mlflow
import mlflow.sklearn

from src.utils.config import settings
from src.utils.logger import logger


class ModelTrainer:
    """
    Trains, cross-validates, evaluates, and serializes 5 machine learning models
    with seamless MLflow experiment tracking.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = self._init_models()
        self.trained_models: Dict[str, Any] = {}
        self.cv_results: Dict[str, Dict[str, Any]] = {}
        self._setup_mlflow()

    def _init_models(self) -> Dict[str, Any]:
        """Initializes the standard dictionary of 5 classification models."""
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )

        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=1.56,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='logloss',
            verbosity=0
        )

        models = {
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state
            ),
            "decision_tree": DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=20,
                class_weight='balanced',
                random_state=self.random_state
            ),
            "random_forest": rf,
            "xgboost": xgb,
            "svm": SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                class_weight='balanced',
                random_state=self.random_state
            )
        }
        return models

    def _setup_mlflow(self):
        """Initializes MLflow tracking URI and experiment."""
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
            logger.info(f"MLflow initialized. URI: {settings.MLFLOW_TRACKING_URI}, Experiment: {settings.MLFLOW_EXPERIMENT_NAME}")
        except Exception as e:
            logger.warning(f"Could not initialize MLflow tracking: {e}")

    def cross_validate_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Runs 5-fold Stratified Cross-Validation on all 5 models."""
        logger.info(f"Starting {cv}-fold cross-validation for all {len(self.models)} models...")
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scoring = ['accuracy', 'f1', 'roc_auc', 'matthews_corrcoef']

        for name, model in self.models.items():
            logger.info(f"Cross-validating model: {name}")
            scores = cross_validate(
                model, X_train, y_train,
                cv=skf,
                scoring=scoring,
                n_jobs=-1
            )
            self.cv_results[name] = {
                "cv_accuracy_mean": float(np.mean(scores['test_accuracy'])),
                "cv_accuracy_std": float(np.std(scores['test_accuracy'])),
                "cv_f1_mean": float(np.mean(scores['test_f1'])),
                "cv_f1_std": float(np.std(scores['test_f1'])),
                "cv_roc_auc_mean": float(np.mean(scores['test_roc_auc'])),
                "cv_roc_auc_std": float(np.std(scores['test_roc_auc'])),
                "cv_mcc_mean": float(np.mean(scores['test_matthews_corrcoef'])),
                "cv_mcc_std": float(np.std(scores['test_matthews_corrcoef'])),
            }
            logger.info(f"[{name}] CV MCC: {self.cv_results[name]['cv_mcc_mean']:.4f} (+/- {self.cv_results[name]['cv_mcc_std']:.4f})")

        return self.cv_results

    def train_single(
        self,
        name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        use_mlflow: bool = False
    ) -> Any:
        """Trains a single model by name and logs parameters and evaluation metrics to MLflow."""
        if name not in self.models:
            raise ValueError(f"Unknown model name: '{name}'. Available: {list(self.models.keys())}")

        logger.info(f"Fitting model: {name}")
        model = self.models[name]

        if use_mlflow:
            with mlflow.start_run(run_name=name):
                try:
                    if hasattr(model, 'get_params'):
                        mlflow.log_params({k: v for k, v in model.get_params().items() if isinstance(v, (int, float, str, bool))})
                except Exception as e:
                    logger.debug(f"Could not log all params for {name}: {e}")

                model.fit(X_train, y_train)

                # Log cross validation metrics if available
                if name in self.cv_results:
                    mlflow.log_metrics(self.cv_results[name])

                # Log test metrics if X_test and y_test are provided
                if X_test is not None and y_test is not None:
                    from src.models.evaluator import ModelEvaluator
                    evaluator = ModelEvaluator()
                    metrics = evaluator.evaluate_single(model, X_test, y_test)
                    mlflow.log_metrics(metrics)

                try:
                    mlflow.sklearn.log_model(model, name)
                except Exception as e:
                    logger.warning(f"Failed to log model artifact in MLflow: {e}")

                self.trained_models[name] = model
                return model
        else:
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            return model

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        use_mlflow: bool = True
    ) -> Dict[str, Any]:
        """Trains all 5 models sequentially, evaluates them, and logs them to MLflow."""
        logger.info(f"Training all {len(self.models)} models on {X_train.shape[0]} samples...")

        for name in self.models.keys():
            self.train_single(name, X_train, y_train, X_test=X_test, y_test=y_test, use_mlflow=use_mlflow)

        logger.info("All 5 models trained successfully.")
        return self.trained_models

    def save_all(self, models: Optional[Dict[str, Any]] = None, path: Optional[str] = None) -> Dict[str, str]:
        """Saves all trained models as individual pickle files in the models directory."""
        target_dir = path or settings.MODELS_PATH
        os.makedirs(target_dir, exist_ok=True)
        models_to_save = models or self.trained_models
        saved_paths = {}

        for name, model in models_to_save.items():
            model_file = os.path.join(target_dir, f"{name}.pkl")
            joblib.dump(model, model_file)
            saved_paths[name] = model_file
            logger.info(f"Saved model '{name}' to {model_file}")

        return saved_paths

    def save_single(self, name: str, model: Any, path: Optional[str] = None) -> str:
        """Saves a single model to disk."""
        target_dir = path or settings.MODELS_PATH
        os.makedirs(target_dir, exist_ok=True)
        model_file = os.path.join(target_dir, f"{name}.pkl")
        joblib.dump(model, model_file)
        logger.info(f"Saved single model '{name}' to {model_file}")
        return model_file

    @staticmethod
    def load_all(path: Optional[str] = None) -> Dict[str, Any]:
        """Loads all available models from the models directory."""
        target_dir = path or settings.MODELS_PATH
        loaded_models = {}
        model_names = [
            "logistic_regression", "decision_tree", "random_forest",
            "xgboost", "svm"
        ]

        for name in model_names:
            model_file = os.path.join(target_dir, f"{name}.pkl")
            if os.path.exists(model_file):
                loaded_models[name] = joblib.load(model_file)
                logger.info(f"Loaded model '{name}' from {model_file}")
            else:
                logger.warning(f"Model file not found: {model_file}")

        return loaded_models

    @staticmethod
    def load_single(name: str, path: Optional[str] = None) -> Any:
        """Loads a single model from the models directory."""
        target_dir = path or settings.MODELS_PATH
        model_file = os.path.join(target_dir, f"{name}.pkl")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model '{name}' not found at: {model_file}")
        return joblib.load(model_file)
