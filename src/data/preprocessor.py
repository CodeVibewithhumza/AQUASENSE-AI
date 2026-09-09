"""Preprocessing pipeline including Grouped Median Imputation, IQR Outlier Capping, and Scaling."""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

from src.data.feature_engineer import WaterFeatureEngineer
from src.utils.config import settings, ALL_FEATURE_NAMES, FEATURE_NAMES
from src.utils.logger import logger


class GroupedMedianImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values using per-class medians during training (when target is available),
    and falls back to overall training medians during inference/testing.
    """

    def __init__(self, features_to_impute: Optional[List[str]] = None):
        self.features_to_impute = features_to_impute or ['ph', 'Sulfate', 'Trihalomethanes']
        self.class_medians_: Dict[int, Dict[str, float]] = {}
        self.global_medians_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Learns per-class and global medians for all columns."""
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Compute global medians for all columns
        for col in df.columns:
            self.global_medians_[col] = float(df[col].median(skipna=True))

        # If y is provided, compute class-specific medians
        if y is not None:
            y_series = pd.Series(y).reset_index(drop=True)
            df_reset = df.reset_index(drop=True)
            for cls in y_series.unique():
                cls_mask = (y_series == cls)
                self.class_medians_[int(cls)] = {}
                for col in df.columns:
                    cls_median = df_reset.loc[cls_mask, col].median(skipna=True)
                    # Fallback to global median if all values in class are NaN
                    if pd.isna(cls_median):
                        cls_median = self.global_medians_[col]
                    self.class_medians_[int(cls)][col] = float(cls_median)

        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Applies imputation using computed medians."""
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        if y is not None and self.class_medians_:
            y_series = pd.Series(y).reset_index(drop=True)
            df_reset = df.reset_index(drop=True)
            for cls, medians in self.class_medians_.items():
                cls_mask = (y_series == cls)
                for col in df.columns:
                    if col in medians:
                        df_reset.loc[cls_mask & df_reset[col].isna(), col] = medians[col]
            df = df_reset
        else:
            for col in df.columns:
                if col in self.global_medians_:
                    df[col] = df[col].fillna(self.global_medians_[col])

        # Fill any remaining NaNs with 0 as safe fallback
        df = df.fillna(0.0)
        return df


class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Caps numeric feature outliers using the Interquartile Range (IQR) method:
    Lower Bound = Q1 - 1.5 * IQR
    Upper Bound = Q3 + 1.5 * IQR
    Fitted exclusively on training data.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_: Dict[str, float] = {}
        self.upper_bounds_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Computes IQR bounds for each column."""
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col in df.columns:
            q1 = float(df[col].quantile(0.25))
            q3 = float(df[col].quantile(0.75))
            iqr = q3 - q1
            self.lower_bounds_[col] = q1 - (self.factor * iqr)
            self.upper_bounds_[col] = q3 + (self.factor * iqr)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clips feature values between lower and upper bounds."""
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col in df.columns:
            if col in self.lower_bounds_ and col in self.upper_bounds_:
                df[col] = df[col].clip(
                    lower=self.lower_bounds_[col],
                    upper=self.upper_bounds_[col]
                )
        return df


class WaterQualityPreprocessor:
    """
    Complete end-to-end preprocessing pipeline for water quality potability data:
    1. Feature Engineering: calculates domain interaction ratios.
    2. Grouped Median Imputation: handles missing values without data leakage.
    3. IQR Capping: handles extreme outliers cleanly.
    4. RobustScaler: scales features robustly against remaining variances.
    """

    def __init__(self):
        self.feature_engineer = WaterFeatureEngineer()
        self.imputer = GroupedMedianImputer()
        self.capper = IQRCapper(factor=1.5)
        self.scaler = RobustScaler()
        self.feature_names_in_: List[str] = FEATURE_NAMES
        self.feature_names_out_: List[str] = ALL_FEATURE_NAMES
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "WaterQualityPreprocessor":
        """Fits all preprocessing stages sequentially on the training dataset."""
        logger.info("Fitting WaterQualityPreprocessor pipeline...")
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)

        # 1. Feature Engineering
        df_fe = self.feature_engineer.fit_transform(df, y)

        # 2. Imputation
        df_imp = self.imputer.fit_transform(df_fe, y)

        # 3. Outlier Capping
        df_cap = self.capper.fit_transform(df_imp, y)

        # 4. Scaling
        self.scaler.fit(df_cap)

        self.feature_names_out_ = df_fe.columns.tolist()
        self.is_fitted_ = True
        logger.info(f"Preprocessor successfully fitted. Output feature count: {len(self.feature_names_out_)}")
        return self

    def transform(self, X: pd.DataFrame, return_df: bool = True) -> Any:
        """Transforms new/test data using the fitted pipeline."""
        if not self.is_fitted_:
            raise RuntimeError("WaterQualityPreprocessor must be fitted before transforming data.")

        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_[:X.shape[1]])

        # 1. Feature Engineering
        df_fe = self.feature_engineer.transform(df)

        # 2. Imputation (inference mode without target label)
        df_imp = self.imputer.transform(df_fe)

        # 3. Outlier Capping
        df_cap = self.capper.transform(df_imp)

        # 4. Robust Scaling
        scaled_array = self.scaler.transform(df_cap)

        if return_df:
            return pd.DataFrame(scaled_array, columns=self.feature_names_out_, index=df.index)
        return scaled_array

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, return_df: bool = True) -> Any:
        """Fits the pipeline and transforms the training data."""
        self.fit(X, y)
        return self.transform(X, return_df=return_df)

    def apply_smote(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Applies SMOTE (Synthetic Minority Oversampling Technique) to the preprocessed training data.
        MUST only be applied to training data to avoid data leakage.
        """
        logger.info(f"Applying SMOTE balancing (initial distribution: {pd.Series(y).value_counts().to_dict()})")
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        if isinstance(X, pd.DataFrame):
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=getattr(y, 'name', 'Potability'))

        logger.info(f"SMOTE applied successfully (balanced distribution: {y_resampled.value_counts().to_dict()})")
        return X_resampled, y_resampled

    def save(self, path: Optional[str] = None) -> str:
        """Serializes the fitted preprocessor to disk using joblib."""
        save_path = path or os.path.join(settings.MODELS_PATH, "preprocessor.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self, save_path)
        logger.info(f"Preprocessor serialized to: {save_path}")
        return save_path

    @classmethod
    def load(cls, path: Optional[str] = None) -> "WaterQualityPreprocessor":
        """Deserializes a fitted preprocessor from disk."""
        load_path = path or os.path.join(settings.MODELS_PATH, "preprocessor.pkl")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Preprocessor pickle not found at: {load_path}")
        logger.info(f"Loading preprocessor from: {load_path}")
        return joblib.load(load_path)
