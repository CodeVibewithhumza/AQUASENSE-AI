"""Data loading and schema validation for water potability dataset."""
import os
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import settings, FEATURE_NAMES
from src.utils.logger import logger


class DataLoader:
    """Handles loading, schema verification, summary statistics, and splitting of the dataset."""

    EXPECTED_FEATURES = FEATURE_NAMES
    TARGET_COLUMN = "Potability"
    ALL_COLUMNS = EXPECTED_FEATURES + [TARGET_COLUMN]

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or settings.DATA_PATH

    def load(self, path: Optional[str] = None) -> pd.DataFrame:
        """Loads CSV dataset and validates column schema."""
        target_path = path or self.data_path
        if not os.path.exists(target_path):
            logger.error(f"Dataset file not found at: {target_path}")
            raise FileNotFoundError(f"Dataset not found at path: {target_path}")

        logger.info(f"Loading dataset from: {target_path}")
        df = pd.read_csv(target_path)
        self.validate_schema(df)
        logger.info(f"Successfully loaded dataset with shape: {df.shape}")
        return df

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Validates that all expected features and target column are present."""
        missing_cols = [col for col in self.ALL_COLUMNS if col not in df.columns]
        if missing_cols:
            error_msg = f"Dataset schema validation failed. Missing required columns: {missing_cols}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate types
        for col in self.EXPECTED_FEATURES:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Column '{col}' must be numeric, got {df[col].dtype}")

        if not pd.api.types.is_numeric_dtype(df[self.TARGET_COLUMN]):
            raise TypeError(f"Target column '{self.TARGET_COLUMN}' must be numeric, got {df[self.TARGET_COLUMN].dtype}")

    def get_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates dataset summary metrics including missing values and class distribution."""
        missing_counts = df.isnull().sum().to_dict()
        missing_percentages = (df.isnull().mean() * 100).round(2).to_dict()

        target_counts = df[self.TARGET_COLUMN].value_counts().to_dict() if self.TARGET_COLUMN in df.columns else {}
        target_percentages = (df[self.TARGET_COLUMN].value_counts(normalize=True) * 100).round(2).to_dict() if self.TARGET_COLUMN in df.columns else {}

        summary = {
            "shape": df.shape,
            "rows": len(df),
            "columns": len(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_counts": missing_counts,
            "missing_percentages": missing_percentages,
            "class_distribution": target_counts,
            "class_distribution_pct": target_percentages,
            "duplicate_rows": int(df.duplicated().sum())
        }
        return summary

    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Performs a stratified train/test split on the dataset."""
        self.validate_schema(df)
        X = df[self.EXPECTED_FEATURES].copy()
        y = df[self.TARGET_COLUMN].copy()

        logger.info(f"Splitting data: test_size={test_size}, random_state={random_state}, stratified=True")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        logger.info(f"Train split shape: {X_train.shape}, Test split shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
