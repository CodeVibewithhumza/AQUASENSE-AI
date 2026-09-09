"""LIME (Local Interpretable Model-agnostic Explanations) for local fidelity analysis."""
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from src.utils.config import COLORS, ALL_FEATURE_NAMES
from src.utils.visualization import set_plot_style
from src.utils.logger import logger


class LIMEExplainer:
    """
    Constructs local sparse linear surrogate models to explain single predictions
    and aggregate local feature attributions.
    """

    def __init__(self, random_state: int = 42):
        self.explainer: Optional[lime.lime_tabular.LimeTabularExplainer] = None
        self.feature_names: List[str] = ALL_FEATURE_NAMES
        self.class_names: List[str] = ['Not Potable', 'Potable']
        self.random_state = random_state

    def fit(
        self,
        X_train: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ) -> "LIMEExplainer":
        """Initializes the LimeTabularExplainer using the training distribution."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = feature_names or X_train.columns.tolist()
            train_arr = X_train.values
        else:
            train_arr = np.asarray(X_train)
            self.feature_names = feature_names or [f"feature_{i}" for i in range(train_arr.shape[1])]

        if class_names:
            self.class_names = class_names

        logger.info(f"Initializing LimeTabularExplainer with {len(self.feature_names)} features...")
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=train_arr,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification',
            random_state=self.random_state
        )
        return self

    def explain_instance(self, model: Any, x: Any, num_features: int = 10) -> Any:
        """Explains a single test sample instance."""
        if self.explainer is None:
            raise RuntimeError("LIMEExplainer must be fitted before explaining instances.")

        if isinstance(x, pd.DataFrame):
            x_arr = x.iloc[0].values
        elif isinstance(x, pd.Series):
            x_arr = x.values
        else:
            x_arr = np.asarray(x).ravel()

        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else lambda d: np.column_stack([1 - model.predict(d), model.predict(d)])

        exp = self.explainer.explain_instance(
            data_row=x_arr,
            predict_fn=predict_fn,
            num_features=num_features,
            labels=(1,)
        )
        return exp

    def get_feature_weights(self, model: Any, x: Any, num_features: int = 10) -> Dict[str, float]:
        """Returns clean dictionary mapping feature names to their LIME attribution weights."""
        exp = self.explain_instance(model, x, num_features=num_features)
        exp_list = exp.as_list(label=1)
        weights = {}

        for rule, weight in exp_list:
            # Match rule condition back to base feature name
            matched_feat = rule
            for feat in self.feature_names:
                if feat in rule:
                    matched_feat = feat
                    break
            weights[matched_feat] = float(weight)

        return weights

    def explain_batch(self, model: Any, X: pd.DataFrame, n_samples: int = 15) -> List[Any]:
        """Explains multiple samples sequentially."""
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names)
        sample_size = min(n_samples, len(df))
        explanations = []

        logger.info(f"Generating LIME batch explanations for {sample_size} samples...")
        for i in range(sample_size):
            row = df.iloc[i]
            exp = self.explain_instance(model, row, num_features=len(self.feature_names))
            explanations.append(exp)

        return explanations

    def get_feature_ranking(self, model: Any, X: pd.DataFrame, n_samples: int = 30) -> List[Tuple[str, float]]:
        """Computes average global feature ranking by aggregating absolute LIME weights across samples."""
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names)
        sample_size = min(n_samples, len(df))
        feature_scores = defaultdict(list)

        logger.info(f"Aggregating LIME feature importance across {sample_size} sample points...")
        for i in range(sample_size):
            row = df.iloc[i]
            weights = self.get_feature_weights(model, row, num_features=len(self.feature_names))
            for feat, wt in weights.items():
                feature_scores[feat].append(abs(wt))

        ranking = []
        for feat in self.feature_names:
            scores = feature_scores.get(feat, [0.0])
            ranking.append((feat, float(np.mean(scores))))

        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def plot_explanation(self, model: Any, x: Any, idx: int = 0, num_features: int = 10) -> plt.Figure:
        """Generates a styled horizontal bar chart for a single LIME instance explanation."""
        set_plot_style()
        weights = self.get_feature_weights(model, x, num_features=num_features)

        sorted_items = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=False)
        feats = [item[0] for item in sorted_items]
        vals = [item[1] for item in sorted_items]
        bar_colors = [COLORS['safe'] if v >= 0 else COLORS['danger'] for v in vals]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(feats, vals, color=bar_colors, edgecolor=COLORS['border'])
        ax.axvline(0, color=COLORS['text_muted'], linestyle='--', alpha=0.6)
        ax.set_title(f"LIME Local Explanation (Sample #{idx})", fontsize=13, color=COLORS['accent'], fontweight='bold')
        ax.set_xlabel("LIME Weight (Contribution to Potability)", fontsize=11, color=COLORS['text_primary'])
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        return fig
