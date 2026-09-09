"""Voting and stacking ensemble constructors for AquaSense AI."""
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Optional, List, Tuple, Any
from src.utils.logger import logger


def create_voting_ensemble(
    rf_model: Optional[RandomForestClassifier] = None,
    xgb_model: Optional[XGBClassifier] = None,
    lgbm_model: Optional[LGBMClassifier] = None,
    voting: str = "soft",
    random_state: int = 42
) -> VotingClassifier:
    """
    Creates a soft-voting ensemble combining Random Forest, XGBoost, and LightGBM.
    Soft voting averages predicted class probabilities across all estimators.
    """
    if rf_model is None:
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )

    if xgb_model is None:
        xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=1.56,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='logloss',
            verbosity=0
        )

    if lgbm_model is None:
        lgbm_model = LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )

    estimators: List[Tuple[str, Any]] = [
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('lgbm', lgbm_model)
    ]

    logger.info(f"Creating VotingClassifier ensemble with {len(estimators)} estimators (voting='{voting}')")
    ensemble = VotingClassifier(
        estimators=estimators,
        voting=voting
    )
    return ensemble
