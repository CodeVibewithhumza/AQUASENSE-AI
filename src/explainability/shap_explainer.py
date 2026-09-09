"""SHAP (SHapley Additive exPlanations) analyzer for global and local interpretability."""
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils.config import COLORS, ALL_FEATURE_NAMES
from src.utils.visualization import set_plot_style
from src.utils.logger import logger


class SHAPExplainer:
    """
    Computes game-theoretic Shapley attributions for tree models and general estimators,
    producing global summary plots, local waterfalls, and feature rankings.
    """

    def __init__(self):
        self.explainer: Optional[Any] = None
        self.feature_names: List[str] = ALL_FEATURE_NAMES
        self.is_tree_model: bool = False
        self.model: Optional[Any] = None

    def fit(self, model: Any, X_train: pd.DataFrame) -> "SHAPExplainer":
        """Initializes TreeExplainer or KernelExplainer depending on model architecture."""
        self.model = model
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_arr = X_train.values
        else:
            X_arr = np.asarray(X_train)

        # Determine if model is tree-based
        tree_types = (RandomForestClassifier, DecisionTreeClassifier, XGBClassifier)
        if isinstance(model, tree_types) or hasattr(model, "tree_"):
            try:
                logger.info("Initializing shap.TreeExplainer for tree model...")
                self.explainer = shap.TreeExplainer(model)
                self.is_tree_model = True
            except Exception as e:
                logger.warning(f"TreeExplainer failed ({e}), falling back to generic Explainer...")
                background = shap.sample(X_arr, min(50, len(X_arr)))
                self.explainer = shap.Explainer(model.predict_proba, background)
                self.is_tree_model = False
        else:
            logger.info("Initializing shap.Explainer / KernelExplainer for non-tree model...")
            background = shap.sample(X_arr, min(50, len(X_arr)))
            self.explainer = shap.Explainer(model.predict_proba, background)
            self.is_tree_model = False

        return self

    def _extract_class1_values(self, shap_obj: Any) -> Any:
        """Extracts SHAP values corresponding to the positive 'Potable' class (index 1)."""
        if isinstance(shap_obj, shap.Explanation):
            if len(shap_obj.values.shape) == 3 and shap_obj.values.shape[2] == 2:
                # Shape (N, num_features, 2)
                exp = shap.Explanation(
                    values=shap_obj.values[:, :, 1],
                    base_values=shap_obj.base_values[:, 1] if len(shap_obj.base_values.shape) > 1 else shap_obj.base_values,
                    data=shap_obj.data,
                    feature_names=self.feature_names
                )
                return exp
            return shap_obj
        elif isinstance(shap_obj, list) and len(shap_obj) == 2:
            return shap_obj[1]
        elif isinstance(shap_obj, np.ndarray):
            if len(shap_obj.shape) == 3 and shap_obj.shape[2] == 2:
                return shap_obj[:, :, 1]
            return shap_obj
        return shap_obj

    def global_explanation(self, X: pd.DataFrame) -> shap.Explanation:
        """Computes global SHAP Explanation object for a dataset."""
        if self.explainer is None:
            raise RuntimeError("SHAPExplainer must be fitted before computing explanations.")

        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names)
        raw_explanation = self.explainer(X_df)
        return self._extract_class1_values(raw_explanation)

    def local_explanation(self, x: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes local SHAP attributions for a single sample instance.
        Returns:
            dict with 'shap_values' mapping, 'base_value', and 'expected_value'.
        """
        if self.explainer is None:
            raise RuntimeError("SHAPExplainer must be fitted before computing explanations.")

        x_df = x if isinstance(x, pd.DataFrame) else pd.DataFrame([x], columns=self.feature_names)
        explanation = self.global_explanation(x_df)

        if isinstance(explanation, shap.Explanation):
            vals = explanation.values[0]
            base_val = float(explanation.base_values[0]) if hasattr(explanation.base_values, '__len__') else float(explanation.base_values)
        else:
            vals = explanation[0]
            base_val = 0.5

        shap_dict = {feat: float(vals[i]) for i, feat in enumerate(self.feature_names[:len(vals)])}

        # Partition top factors
        sorted_factors = sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)
        top_for = [k for k, v in sorted_factors if v > 0][:5]
        top_against = [k for k, v in sorted_factors if v < 0][:5]

        return {
            "shap_values": shap_dict,
            "shap_base_value": base_val,
            "top_factors_for_potable": top_for,
            "top_factors_against_potable": top_against
        }

    def get_feature_ranking(self, shap_values: Any, feature_names: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Computes global feature ranking by mean absolute SHAP value."""
        feats = feature_names or self.feature_names
        if isinstance(shap_values, shap.Explanation):
            vals = np.abs(shap_values.values).mean(axis=0)
        elif isinstance(shap_values, np.ndarray):
            vals = np.abs(shap_values).mean(axis=0)
        else:
            vals = np.zeros(len(feats))

        ranking = [(feats[i], float(vals[i])) for i in range(min(len(feats), len(vals)))]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def plot_beeswarm(self, shap_values: Any, X: pd.DataFrame, max_display: int = 12) -> plt.Figure:
        """Generates a SHAP Beeswarm summary plot."""
        set_plot_style()
        fig, ax = plt.subplots(figsize=(10, 7))
        plt.sca(ax)

        if isinstance(shap_values, shap.Explanation):
            shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
        else:
            shap.summary_plot(shap_values, X, max_display=max_display, show=False)

        plt.title("SHAP Global Feature Impact Distribution (Beeswarm)", fontsize=13, color=COLORS['accent'], fontweight='bold')
        plt.tight_layout()
        return plt.gcf()

    def plot_bar(self, shap_values: Any, max_display: int = 12) -> plt.Figure:
        """Generates a SHAP mean absolute value bar chart."""
        set_plot_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.sca(ax)

        if isinstance(shap_values, shap.Explanation):
            shap.plots.bar(shap_values, max_display=max_display, show=False)
        else:
            shap.summary_plot(shap_values, plot_type="bar", max_display=max_display, show=False)

        plt.title("SHAP Global Feature Importance (Mean |SHAP Value|)", fontsize=13, color=COLORS['accent'], fontweight='bold')
        plt.tight_layout()
        return plt.gcf()

    def plot_waterfall(self, shap_values: shap.Explanation, idx: int = 0, max_display: int = 10) -> plt.Figure:
        """Generates a SHAP Waterfall attribution plot for an individual prediction."""
        set_plot_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.sca(ax)

        if isinstance(shap_values, shap.Explanation):
            shap.plots.waterfall(shap_values[idx], max_display=max_display, show=False)
        else:
            logger.warning("Waterfall plot requires a shap.Explanation object.")

        plt.title(f"SHAP Local Waterfall Attribution (Sample #{idx})", fontsize=13, color=COLORS['accent'], fontweight='bold')
        plt.tight_layout()
        return plt.gcf()

    def plot_dependence(
        self,
        shap_values: Any,
        X: pd.DataFrame,
        feature: str,
        interaction_feature: Optional[str] = None
    ) -> plt.Figure:
        """Plots SHAP feature dependence with interaction coloring."""
        set_plot_style()
        fig, ax = plt.subplots(figsize=(9, 6))
        plt.sca(ax)

        vals = shap_values.values if isinstance(shap_values, shap.Explanation) else shap_values
        shap.dependence_plot(
            feature,
            vals,
            X,
            interaction_index=interaction_feature,
            show=False,
            ax=ax
        )
        plt.title(f"SHAP Dependence: {feature}", fontsize=13, color=COLORS['accent'], fontweight='bold')
        plt.tight_layout()
        return plt.gcf()
