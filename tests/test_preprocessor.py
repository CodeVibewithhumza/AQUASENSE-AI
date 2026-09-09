"""Unit tests for data loader, feature engineering, and preprocessing pipeline."""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from src.data.loader import DataLoader
from src.data.feature_engineer import WaterFeatureEngineer
from src.data.preprocessor import (
    WaterQualityPreprocessor,
    GroupedMedianImputer,
    IQRCapper
)
from src.utils.config import FEATURE_NAMES, ALL_FEATURE_NAMES


@pytest.fixture
def sample_raw_data():
    """Generates a small synthetic DataFrame with raw feature columns and missing values."""
    np.random.seed(42)
    n = 100
    data = {
        "ph": np.random.uniform(5.0, 9.0, n),
        "Hardness": np.random.uniform(100.0, 300.0, n),
        "Solids": np.random.uniform(10000.0, 40000.0, n),
        "Chloramines": np.random.uniform(3.0, 10.0, n),
        "Sulfate": np.random.uniform(200.0, 450.0, n),
        "Conductivity": np.random.uniform(200.0, 700.0, n),
        "Organic_carbon": np.random.uniform(5.0, 25.0, n),
        "Trihalomethanes": np.random.uniform(20.0, 100.0, n),
        "Turbidity": np.random.uniform(1.5, 6.5, n),
        "Potability": np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    }
    # Insert NaNs in ph, Sulfate, Trihalomethanes
    data["ph"][0:10] = np.nan
    data["Sulfate"][10:25] = np.nan
    data["Trihalomethanes"][25:35] = np.nan

    return pd.DataFrame(data)


def test_data_loader_validation(sample_raw_data):
    """Verifies DataLoader schema validation catches missing columns."""
    dl = DataLoader()
    dl.validate_schema(sample_raw_data)

    invalid_df = sample_raw_data.drop(columns=["ph"])
    with pytest.raises(ValueError):
        dl.validate_schema(invalid_df)


def test_data_loader_split(sample_raw_data):
    """Verifies stratified split proportions and dimensions."""
    dl = DataLoader()
    X_train, X_test, y_train, y_test = dl.split(sample_raw_data, test_size=0.20, random_state=42)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
    assert X_train.shape[1] == 9


def test_feature_engineer(sample_raw_data):
    """Verifies that all 6 interaction features are created with valid values."""
    X = sample_raw_data[FEATURE_NAMES]
    fe = WaterFeatureEngineer()
    X_fe = fe.fit_transform(X)

    assert X_fe.shape[1] == 15
    for feat in [
        'ph_hardness_ratio', 'tds_conductivity_ratio',
        'chloramine_organic_interaction', 'trihalomethane_risk',
        'hardness_sulfate_ratio', 'overall_contamination_index'
    ]:
        assert feat in X_fe.columns
        assert not np.isinf(X_fe[feat].dropna()).any()


def test_grouped_median_imputer(sample_raw_data):
    """Verifies that GroupedMedianImputer fills all missing values without error."""
    X = sample_raw_data[FEATURE_NAMES]
    y = sample_raw_data["Potability"]

    imputer = GroupedMedianImputer()
    imputer.fit(X, y)
    X_imp = imputer.transform(X)

    assert X_imp.isnull().sum().sum() == 0


def test_iqr_capper(sample_raw_data):
    """Verifies that IQRCapper constrains extreme outliers."""
    X = sample_raw_data[FEATURE_NAMES].fillna(0)
    capper = IQRCapper(factor=1.5)
    X_cap = capper.fit_transform(X)

    for col in FEATURE_NAMES:
        assert X_cap[col].max() <= capper.upper_bounds_[col] + 1e-5
        assert X_cap[col].min() >= capper.lower_bounds_[col] - 1e-5


def test_full_pipeline_and_smote(sample_raw_data):
    """Verifies end-to-end fit, transform, SMOTE oversampling, and pickle serialization."""
    X = sample_raw_data[FEATURE_NAMES]
    y = sample_raw_data["Potability"]

    pipeline = WaterQualityPreprocessor()
    X_proc = pipeline.fit_transform(X, y)

    assert X_proc.shape[1] == 15
    assert X_proc.isnull().sum().sum() == 0

    # Test SMOTE
    X_bal, y_bal = pipeline.apply_smote(X_proc, y, random_state=42)
    assert len(X_bal) == len(y_bal)
    assert y_bal.value_counts().iloc[0] == y_bal.value_counts().iloc[1]

    # Test save/load
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pipeline.save(tmp_path)
        loaded = WaterQualityPreprocessor.load(tmp_path)
        assert loaded.is_fitted_
        X_test_proc = loaded.transform(X.head(5))
        assert X_test_proc.shape == (5, 15)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
