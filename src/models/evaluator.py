"""Evaluation framework computing 7 classification metrics and producing diagnostic charts."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    log_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from sklearn.model_selection import learning_curve

from src.utils.config import COLORS
from src.utils.visualization import set_plot_style
from src.utils.logger import logger


class ModelEvaluator:
    """
    Computes standard and robust classification metrics across all models
    and provides diagnostic visual analysis.
    """

    def __init__(self):
        set_plot_style()

    def evaluate_single(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Calculates 7 performance metrics for a single model."""
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            df = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-df))
        else:
            y_prob = y_pred

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        mcc = float(matthews_corrcoef(y_test, y_pred))

        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            auc = 0.5

        try:
            ll = float(log_loss(y_test, y_prob))
        except Exception:
            ll = float('nan')

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(auc, 4),
            "mcc": round(mcc, 4),
            "log_loss": round(ll, 4)
        }

    def evaluate_all(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """Evaluates all provided models and returns a benchmark summary DataFrame."""
        logger.info(f"Evaluating {len(models)} models on test dataset ({len(y_test)} samples)...")
        results = {}

        for name, model in models.items():
            results[name] = self.evaluate_single(model, X_test, y_test)
            logger.info(f"[{name}] Acc: {results[name]['accuracy']:.4f} | F1: {results[name]['f1']:.4f} | AUC: {results[name]['roc_auc']:.4f} | MCC: {results[name]['mcc']:.4f}")

        results_df = pd.DataFrame.from_dict(results, orient='index')
        return results_df

    def get_best_model(self, results_df: pd.DataFrame, metric: str = 'mcc') -> str:
        """Identifies and returns the best-performing model name according to a specified metric."""
        if metric not in results_df.columns:
            raise ValueError(f"Metric '{metric}' not found in results. Available: {list(results_df.columns)}")
        best_model_name = str(results_df[metric].idxmax())
        logger.info(f"Best model selected based on {metric.upper()}: '{best_model_name}' (Score: {results_df.loc[best_model_name, metric]:.4f})")
        return best_model_name

    def plot_confusion_matrices(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> plt.Figure:
        """Generates a 2x4 grid of styled confusion matrices for all models."""
        set_plot_style()
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        axes = axes.flatten()

        model_names = list(models.keys())
        for i, name in enumerate(model_names[:8]):
            model = models[name]
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            ax = axes[i]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap=sns.dark_palette(COLORS['accent'], as_cmap=True),
                cbar=False,
                ax=ax,
                xticklabels=["Not Potable (0)", "Potable (1)"],
                yticklabels=["Not Potable (0)", "Potable (1)"]
            )
            ax.set_title(name.replace('_', ' ').title(), color=COLORS['accent'], fontsize=12, fontweight='bold')
            ax.set_xlabel("Predicted Label", color=COLORS['text_muted'], fontsize=10)
            ax.set_ylabel("True Label", color=COLORS['text_muted'], fontsize=10)

        # Hide any unused subplots
        for j in range(len(model_names), len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle("AquaSense AI — Confusion Matrix Benchmark", fontsize=16, color=COLORS['text_primary'], y=0.98)
        plt.tight_layout()
        return fig

    def plot_roc_curves(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> plt.Figure:
        """Plots all model ROC curves on a single high-contrast canvas."""
        set_plot_style()
        fig, ax = plt.subplots(figsize=(10, 7))

        palette = [
            COLORS['accent'], COLORS['safe'], '#FFAA00', '#FF4B4B',
            '#9D4EDD', '#00BBF9', '#F72585', '#4CC9F0'
        ]

        for idx, (name, model) in enumerate(models.items()):
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                df = model.decision_function(X_test)
                y_prob = 1 / (1 + np.exp(-df))
            else:
                continue

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            color = palette[idx % len(palette)]
            ax.plot(fpr, tpr, label=f"{name.replace('_', ' ').title()} (AUC = {auc:.3f})", color=color, linewidth=2)

        ax.plot([0, 1], [0, 1], linestyle='--', color=COLORS['text_muted'], alpha=0.6, label='Random Chance (AUC = 0.500)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, color=COLORS['text_primary'])
        ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12, color=COLORS['text_primary'])
        ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, color=COLORS['accent'], fontweight='bold')
        ax.legend(loc="lower right", facecolor=COLORS['bg_surface'], edgecolor=COLORS['border'])
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_pr_curves(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> plt.Figure:
        """Plots all model Precision-Recall curves on a single canvas."""
        set_plot_style()
        fig, ax = plt.subplots(figsize=(10, 7))

        palette = [
            COLORS['accent'], COLORS['safe'], '#FFAA00', '#FF4B4B',
            '#9D4EDD', '#00BBF9', '#F72585', '#4CC9F0'
        ]

        for idx, (name, model) in enumerate(models.items()):
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                df = model.decision_function(X_test)
                y_prob = 1 / (1 + np.exp(-df))
            else:
                continue

            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            color = palette[idx % len(palette)]
            ax.plot(rec, prec, label=f"{name.replace('_', ' ').title()}", color=color, linewidth=2)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall", fontsize=12, color=COLORS['text_primary'])
        ax.set_ylabel("Precision", fontsize=12, color=COLORS['text_primary'])
        ax.set_title("Precision-Recall (PR) Curve Comparison", fontsize=14, color=COLORS['accent'], fontweight='bold')
        ax.legend(loc="lower left", facecolor=COLORS['bg_surface'], edgecolor=COLORS['border'])
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_model_comparison(self, results_df: pd.DataFrame) -> plt.Figure:
        """Produces a grouped bar chart comparing all models across all primary metrics."""
        set_plot_style()
        metrics_to_plot = ['accuracy', 'f1', 'roc_auc', 'mcc']
        df_plot = results_df[metrics_to_plot].copy()
        df_plot.index = [i.replace('_', ' ').title() for i in df_plot.index]

        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot.plot(
            kind='bar',
            ax=ax,
            color=[COLORS['accent'], COLORS['safe'], '#FFAA00', '#9D4EDD'],
            width=0.8,
            edgecolor=COLORS['border']
        )
        ax.set_title("Model Performance Benchmark Across Primary Metrics", fontsize=14, color=COLORS['accent'], fontweight='bold')
        ax.set_ylabel("Score", fontsize=12, color=COLORS['text_primary'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', color=COLORS['text_primary'])
        ax.set_ylim([0, 1.05])
        ax.legend(title="Metric", facecolor=COLORS['bg_surface'], edgecolor=COLORS['border'])
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_learning_curves(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 5
    ) -> plt.Figure:
        """Plots learning curves comparing training and validation scores against sample size."""
        set_plot_style()
        fig, ax = plt.subplots(figsize=(9, 6))

        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train,
            cv=cv,
            scoring='matthews_corrcoef',
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 6),
            random_state=42
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        ax.plot(train_sizes, train_mean, 'o-', color=COLORS['accent'], label="Training Score (MCC)")
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color=COLORS['accent'])

        ax.plot(train_sizes, test_mean, 'o-', color=COLORS['safe'], label="Cross-Validation Score (MCC)")
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color=COLORS['safe'])

        ax.set_title("Learning Curve Diagnostic (Matthews Correlation Coefficient)", fontsize=13, color=COLORS['accent'], fontweight='bold')
        ax.set_xlabel("Training Set Size", fontsize=11, color=COLORS['text_primary'])
        ax.set_ylabel("MCC Score", fontsize=11, color=COLORS['text_primary'])
        ax.legend(loc="best", facecolor=COLORS['bg_surface'], edgecolor=COLORS['border'])
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        return fig
