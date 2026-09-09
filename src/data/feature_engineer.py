"""Domain-specific feature engineering transformer for water quality metrics."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List
from src.utils.config import FEATURE_NAMES, ALL_FEATURE_NAMES
from src.utils.logger import logger


class WaterFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Constructs domain-specific physicochemical interaction features:
    1. ph_hardness_ratio: pH relative to mineral hardness
    2. tds_conductivity_ratio: Total dissolved solids relative to electrical conductivity
    3. chloramine_organic_interaction: Disinfectant-organic matter interaction proxy
    4. trihalomethane_risk: Disinfection byproduct risk quotient
    5. hardness_sulfate_ratio: Mineral cation-anion balance indicator
    6. overall_contamination_index: Composite normalized contamination index (Solids, Turbidity, Organic_carbon)
    """

    def __init__(self):
        self.min_vals_ = {}
        self.max_vals_ = {}
        self.feature_names_in_ = FEATURE_NAMES
        self.feature_names_out_ = ALL_FEATURE_NAMES

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Learns feature scaling bounds for the composite contamination index."""
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)

        # Learn min/max for normalization of contamination indicators
        for col in ['Solids', 'Turbidity', 'Organic_carbon']:
            if col in df.columns:
                self.min_vals_[col] = float(df[col].min(skipna=True))
                self.max_vals_[col] = float(df[col].max(skipna=True))
            else:
                self.min_vals_[col] = 0.0
                self.max_vals_[col] = 1.0

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies feature transformations and appends engineered columns."""
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=self.feature_names_in_[:X.shape[1]])

        # 1. pH to Hardness Ratio
        df['ph_hardness_ratio'] = df['ph'] / (df['Hardness'] + 1.0)

        # 2. TDS to Conductivity Ratio
        df['tds_conductivity_ratio'] = df['Solids'] / (df['Conductivity'] + 1.0)

        # 3. Chloramine & Organic Carbon Interaction
        df['chloramine_organic_interaction'] = df['Chloramines'] * df['Organic_carbon']

        # 4. Trihalomethane Formation Risk Quotient
        df['trihalomethane_risk'] = df['Trihalomethanes'] / (df['Chloramines'] + 1.0)

        # 5. Hardness to Sulfate Mineral Ratio
        df['hardness_sulfate_ratio'] = df['Hardness'] / (df['Sulfate'] + 1.0)

        # 6. Overall Contamination Index (Mean of MinMax-scaled Solids, Turbidity, Organic_carbon)
        solids_norm = self._minmax_scale(df['Solids'], 'Solids')
        turbidity_norm = self._minmax_scale(df['Turbidity'], 'Turbidity')
        organic_norm = self._minmax_scale(df['Organic_carbon'], 'Organic_carbon')

        df['overall_contamination_index'] = (solids_norm + turbidity_norm + organic_norm) / 3.0

        return df

    def _minmax_scale(self, series: pd.Series, col: str) -> pd.Series:
        min_v = self.min_vals_.get(col, 0.0)
        max_v = self.max_vals_.get(col, 1.0)
        denom = max_v - min_v if max_v != min_v else 1.0
        return (series - min_v) / denom

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        return self.feature_names_out_
