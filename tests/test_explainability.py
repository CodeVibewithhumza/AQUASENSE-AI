"""Unit tests for XAI explainability modules (SHAP, LIME, Permutation, Consistency)."""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.permutation import PermutationImportanceAnalyzer
from src.explainability.consistency import ExplanationConsistencyAnalyzer
from src.utils.config import ALL_FEATURE_NAMES


@pytest.fixture
def fitted_tree_model_and_data():
    """Provides a fitted RandomForestClassifier and small synthetic datasets."""
    np.random.seed(42)
    n = 60
    feats = ALL_FEATURE_NAMES
    X_train = pd.DataFrame(np.random.randn(n, len(feats)), columns=feats)
    y_train = pd.Series(np.random.choice([0, 1], size=n))

    X_test = pd.DataFrame(np.random.randn(20, len(feats)), columns=feats)
    y_test = pd.Series(np.random.choice([0, 1], size=20))

    rf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    rf.fit(X_train, y_train)

    return rf, X_train, X_test, y_test


def test_shap_explainer(fitted_tree_model_and_data):
    """Verifies SHAP global and local attribution calculations."""
    rf, X_train, X_test, _ = fitted_tree_model_and_data

    shap_exp = SHAPExplainer().fit(rf, X_train)
    global_exp = shap_exp.global_explanation(X_test.head(10))
    assert global_exp is not None

    local_res = shap_exp.local_explanation(X_test.head(1))
    assert "shap_values" in local_res
    assert len(local_res["shap_values"]) == len(ALL_FEATURE_NAMES)
    assert "top_factors_for_potable" in local_res
    assert "top_factors_against_potable" in local_res


def test_lime_explainer(fitted_tree_model_and_data):
    """Verifies LIME surrogate explanation and weight extraction."""
    rf, X_train, X_test, _ = fitted_tree_model_and_data

    lime_exp = LIMEExplainer(random_state=42).fit(X_train, feature_names=ALL_FEATURE_NAMES)
    weights = lime_exp.get_feature_weights(rf, X_test.iloc[0], num_features=5)

    assert len(weights) > 0
    ranking = lime_exp.get_feature_ranking(rf, X_test.head(5), n_samples=3)
    assert len(ranking) == len(ALL_FEATURE_NAMES)


def test_permutation_importance(fitted_tree_model_and_data):
    """Verifies permutation feature importance computation."""
    rf, _, X_test, y_test = fitted_tree_model_and_data

    perm = PermutationImportanceAnalyzer(random_state=42)
    res = perm.compute(rf, X_test, y_test, n_repeats=3, scoring='accuracy')

    assert "importances_mean" in res
    assert len(res["importances_mean"]) == len(ALL_FEATURE_NAMES)
    ranking = perm.get_feature_ranking(res, ALL_FEATURE_NAMES)
    assert len(ranking) == len(ALL_FEATURE_NAMES)


def test_consistency_analyzer(fitted_tree_model_and_data):
    """Verifies cross-method Spearman correlation and agreement classification."""
    rf, X_train, X_test, y_test = fitted_tree_model_and_data

    analyzer = ExplanationConsistencyAnalyzer(random_state=42)
    rankings, scores = analyzer.compute_all_rankings(
        rf, X_train, X_test.head(10), y_test.head(10),
        n_lime_samples=3, n_perm_repeats=3
    )

    assert "shap" in rankings
    assert "lime" in rankings
    assert "permutation" in rankings

    corr_df = analyzer.spearman_correlation_matrix(rankings)
    assert corr_df.shape == (3, 3)
    assert corr_df.loc["SHAP", "SHAP"] == 1.0

    agree_df = analyzer.agreement_table(rankings)
    assert "Agreement" in agree_df.columns
    assert set(agree_df["Agreement"].unique()).issubset({"Strong", "Moderate", "Weak"})

    report = analyzer.generate_consistency_report(rankings, corr_df, agree_df)
    assert "Explanation Consistency Assessment" in report
