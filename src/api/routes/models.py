"""Model metadata, benchmark metrics, and global feature importance endpoints."""
from fastapi import APIRouter, Request, HTTPException, status
from typing import Dict, Any

from src.api.models import (
    ModelsListResponse,
    ModelMetricItem,
    FeatureImportanceResponse,
    FeatureImportanceItem
)
from src.database.crud import get_model_runs_sync
from src.utils.logger import logger

router = APIRouter(tags=["Models & Metrics"])


@router.get("/models", response_model=ModelsListResponse)
def get_all_models(request: Request):
    """Returns evaluation benchmark metrics for all trained models."""
    best_model_name = getattr(request.app.state, "best_model_name", "xgboost")
    cached_metrics = getattr(request.app.state, "evaluation_metrics", {})

    # Fallback to DB if not cached in app.state
    if not cached_metrics:
        db_runs = get_model_runs_sync()
        for r in db_runs:
            cached_metrics[r["model_name"]] = r

    items = []
    ALLOWED_MODELS = {"logistic_regression", "decision_tree", "random_forest", "xgboost", "svm"}

    for name, m in cached_metrics.items():
        if name not in ALLOWED_MODELS:
            continue
        items.append(ModelMetricItem(
            name=name,
            accuracy=float(m.get("accuracy", 0.0)),
            precision=float(m.get("precision") or m.get("precision_score", 0.0)),
            recall=float(m.get("recall", 0.0)),
            f1=float(m.get("f1") or m.get("f1_score", 0.0)),
            roc_auc=float(m.get("roc_auc", 0.0)),
            mcc=float(m.get("mcc", 0.0)),
            log_loss=float(m["log_loss"]) if m.get("log_loss") is not None else None,
            is_best=(name == best_model_name)
        ))

    # Sort descending by MCC
    items.sort(key=lambda x: x.mcc, reverse=True)
    return ModelsListResponse(models=items, best_model=best_model_name)


@router.get("/models/{model_name}/metrics", response_model=ModelMetricItem)
def get_single_model_metrics(model_name: str, request: Request):
    """Returns detailed performance metrics for a specific classification model."""
    best_model_name = getattr(request.app.state, "best_model_name", "xgboost")
    cached_metrics = getattr(request.app.state, "evaluation_metrics", {})

    clean_name = model_name.lower().replace("-", "_")
    if clean_name not in cached_metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metrics for model '{model_name}' not found. Available: {list(cached_metrics.keys())}"
        )

    m = cached_metrics[clean_name]
    return ModelMetricItem(
        name=clean_name,
        accuracy=float(m.get("accuracy", 0.0)),
        precision=float(m.get("precision") or m.get("precision_score", 0.0)),
        recall=float(m.get("recall", 0.0)),
        f1=float(m.get("f1") or m.get("f1_score", 0.0)),
        roc_auc=float(m.get("roc_auc", 0.0)),
        mcc=float(m.get("mcc", 0.0)),
        log_loss=float(m["log_loss"]) if m.get("log_loss") is not None else None,
        is_best=(clean_name == best_model_name)
    )


@router.get("/feature-importance", response_model=FeatureImportanceResponse)
def get_global_feature_importance(request: Request):
    """Returns global SHAP feature importance rankings for the top-performing model."""
    best_model_name = getattr(request.app.state, "best_model_name", "xgboost")
    global_rankings = getattr(request.app.state, "global_feature_importance", [])

    items = [
        FeatureImportanceItem(
            feature=feat,
            importance=round(float(score), 4),
            rank=i + 1
        )
        for i, (feat, score) in enumerate(global_rankings)
    ]

    return FeatureImportanceResponse(
        model_name=best_model_name,
        method="SHAP",
        rankings=items
    )
