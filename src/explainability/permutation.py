"""Permutation feature importance computation and error-bar visualizations."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.inspection import permutation_importance

from src.utils.config import COLORS, ALL_FEATURE_NAMES
from src.utils.visualization import set_plot_style
from src.utils.logger import logger


class PermutationImportanceAnalyzer:
    """
    Evaluates global feature impact by measuring metric decay when feature values are permuted.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def compute(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_repeats: int = 30,
        scoring: str = 'matthews_corrcoef'
    ) -> Dict[str, Any]:
        """Runs permutation importance on the test set."""
        logger.info(f"Computing permutation importance (n_repeats={n_repeats}, scoring='{scoring}')...")
        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            scoring=scoring,
            random_state=self.random_state,
            n_jobs=-1
        )

        importances = {
            "importances_mean": result.importances_mean,
            "importances_std": result.importances_std,
            "importances": result.importances
        }
        return importances

    def get_feature_ranking(
        self,
        importances: Dict[str, Any],
        feature_names: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Extracts sorted feature list ordered by mean importance decrease."""
        feats = feature_names or ALL_FEATURE_NAMES
        means = importances["importances_mean"]

        ranking = [(feats[i], float(means[i])) for i in range(min(len(feats), len(means)))]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def plot_importance(
        self,
        importances: Dict[str, Any],
        feature_names: Optional[List[str]] = None,
        max_display: int = 12
    ) -> plt.Figure:
        """Generates a horizontal bar chart with standard deviation error bars."""
        set_plot_style()
        feats = feature_names or ALL_FEATURE_NAMES
        means = importances["importances_mean"]
        stds = importances["importances_std"]

        # Sort features by mean importance
        indices = np.argsort(means)[-max_display:]

        sorted_feats = [feats[i] for i in indices]
        sorted_means = means[indices]
        sorted_stds = stds[indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            sorted_feats,
            sorted_means,
            xerr=sorted_stds,
            color=COLORS['accent'],
            edgecolor=COLORS['border'],
            ecolor=COLORS['text_muted'],
            capsize=4
        )
        ax.set_title("Permutation Feature Importance (MCC Drop ± Std)", fontsize=13, color=COLORS['accent'], fontweight='bold')
        ax.set_xlabel("Mean Decrease in MCC Metric", fontsize=11, color=COLORS['text_primary'])
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        return fig
