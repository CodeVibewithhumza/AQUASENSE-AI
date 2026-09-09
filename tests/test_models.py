"""Unit tests for ML models, evaluator metrics, and probability calibration."""
import pytest
import numpy as np
import pandas as pd

from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.models.calibrator import ModelCalibrator
from src.utils.config import ALL_FEATURE_NAMES


@pytest.fixture
def synthetic_train_test():
    """Generates synthetic processed dataset for fast unit testing."""
    np.random.seed(42)
    n_train = 120
    n_test = 40
    n_feats = len(ALL_FEATURE_NAMES)

    X_train = pd.DataFrame(np.random.randn(n_train, n_feats), columns=ALL_FEATURE_NAMES)
    y_train = pd.Series(np.random.choice([0, 1], size=n_train, p=[0.5, 0.5]))

    X_test = pd.DataFrame(np.random.randn(n_test, n_feats), columns=ALL_FEATURE_NAMES)
    y_test = pd.Series(np.random.choice([0, 1], size=n_test, p=[0.5, 0.5]))

    return X_train, y_train, X_test, y_test


def test_model_trainer_initialization():
    """Verifies that all 5 models are initialized with expected keys."""
    trainer = ModelTrainer(random_state=42)
    expected_models = [
        "logistic_regression", "decision_tree", "random_forest",
        "xgboost", "svm"
    ]
    for m in expected_models:
        assert m in trainer.models
    assert len(trainer.models) == 5


def test_model_trainer_single_and_all(synthetic_train_test):
    """Verifies fitting of individual models and all models."""
    X_train, y_train, X_test, y_test = synthetic_train_test
    trainer = ModelTrainer(random_state=42)

    rf_model = trainer.train_single("random_forest", X_train, y_train, use_mlflow=False)
    preds = rf_model.predict(X_test)
    assert len(preds) == len(X_test)

    # Train mini subset
    dt = trainer.train_single("decision_tree", X_train, y_train, use_mlflow=False)
    assert hasattr(dt, "predict")


def test_model_evaluator_metrics(synthetic_train_test):
    """Verifies evaluation metric calculations."""
    X_train, y_train, X_test, y_test = synthetic_train_test
    trainer = ModelTrainer(random_state=42)
    rf = trainer.train_single("random_forest", X_train, y_train, use_mlflow=False)

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_single(rf, X_test, y_test)

    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "mcc"]:
        assert metric in metrics
        assert isinstance(metrics[metric], float)

    df_eval = evaluator.evaluate_all({"rf": rf}, X_test, y_test)
    assert df_eval.shape[0] == 1
    best_name = evaluator.get_best_model(df_eval, metric="mcc")
    assert best_name == "rf"


def test_model_calibrator_and_labels(synthetic_train_test):
    """Verifies probability calibration and qualitative label mapping."""
    X_train, y_train, X_test, y_test = synthetic_train_test
    trainer = ModelTrainer(random_state=42)
    rf = trainer.train_single("random_forest", X_train, y_train, use_mlflow=False)

    calibrator = ModelCalibrator(method='isotonic', cv=2)
    calibrated = calibrator.calibrate(rf, X_train, y_train)

    probs = calibrated.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)

    # Confidence label tiers
    assert ModelCalibrator.get_confidence_label(0.55) == "Low"
    assert ModelCalibrator.get_confidence_label(0.68) == "Moderate"
    assert ModelCalibrator.get_confidence_label(0.82) == "High"
    assert ModelCalibrator.get_confidence_label(0.95) == "Very High"
