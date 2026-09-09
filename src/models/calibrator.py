"""Probability calibration and uncertainty quantification for water potability predictions."""
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Any
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from src.utils.config import settings, COLORS
from src.utils.visualization import set_plot_style
from src.utils.logger import logger


class ModelCalibrator:
    """
    Calibrates model output probabilities to ensure true empirical confidence,
    generates reliability diagrams, and maps probabilities to qualitative confidence tiers.
    """

    def __init__(self, method: str = 'isotonic', cv: int = 5):
        self.method = method
        self.cv = cv
        set_plot_style()

    def calibrate(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> CalibratedClassifierCV:
        """Fits an isotonic or sigmoid probability calibrator over the base classifier."""
        logger.info(f"Calibrating model using method='{self.method}', cv={self.cv}...")
        calibrated_model = CalibratedClassifierCV(
            estimator=model,
            method=self.method,
            cv=self.cv
        )
        calibrated_model.fit(X_train, y_train)
        logger.info("Model calibration complete.")
        return calibrated_model

    def plot_calibration_curve(
        self,
        calibrated_model: Any,
        uncalibrated_model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> plt.Figure:
        """Plots reliability diagrams (calibration curves) for both uncalibrated and calibrated models."""
        set_plot_style()
        fig, ax = plt.subplots(figsize=(8, 6))

        # Uncalibrated curve
        prob_uncal = uncalibrated_model.predict_proba(X_test)[:, 1]
        prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, prob_uncal, n_bins=10)

        # Calibrated curve
        prob_cal = calibrated_model.predict_proba(X_test)[:, 1]
        prob_true_cal, prob_pred_cal = calibration_curve(y_test, prob_cal, n_bins=10)

        ax.plot([0, 1], [0, 1], linestyle='--', color=COLORS['text_muted'], alpha=0.7, label="Perfectly Calibrated")
        ax.plot(prob_pred_uncal, prob_true_uncal, marker='s', color=COLORS['danger'], label="Uncalibrated Model", linewidth=2)
        ax.plot(prob_pred_cal, prob_true_cal, marker='o', color=COLORS['safe'], label=f"Calibrated ({self.method.title()})", linewidth=2)

        ax.set_xlabel("Mean Predicted Probability", fontsize=11, color=COLORS['text_primary'])
        ax.set_ylabel("Empirical True Fraction of Positives", fontsize=11, color=COLORS['text_primary'])
        ax.set_title("Probability Calibration Reliability Diagram", fontsize=13, color=COLORS['accent'], fontweight='bold')
        ax.legend(loc="upper left", facecolor=COLORS['bg_surface'], edgecolor=COLORS['border'])
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def get_confidence_label(probability: float) -> str:
        """
        Maps a prediction probability (relative to predicted class certainty) to a qualitative label:
        - < 0.60       -> 'Low'
        - 0.60 - 0.75  -> 'Moderate'
        - 0.75 - 0.90  -> 'High'
        - > 0.90       -> 'Very High'
        """
        # If probability is for class 1, certainty is max(p, 1-p)
        certainty = probability if probability >= 0.5 else (1.0 - probability)

        if certainty < 0.60:
            return "Low"
        elif certainty < 0.75:
            return "Moderate"
        elif certainty < 0.90:
            return "High"
        else:
            return "Very High"

    @staticmethod
    def save(calibrated_model: Any, path: Optional[str] = None) -> str:
        """Saves calibrated model to disk."""
        target_path = path or os.path.join(settings.MODELS_PATH, "calibrated_best_model.pkl")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        joblib.dump(calibrated_model, target_path)
        logger.info(f"Calibrated model saved to: {target_path}")
        return target_path

    @staticmethod
    def load(path: Optional[str] = None) -> Any:
        """Loads calibrated model from disk."""
        target_path = path or os.path.join(settings.MODELS_PATH, "calibrated_best_model.pkl")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Calibrated model not found at: {target_path}")
        return joblib.load(target_path)
