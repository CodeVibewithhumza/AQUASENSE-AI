"""CRUD operations for asynchronous and synchronous database interaction."""
import json
from typing import Dict, List, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from src.database.models import Prediction, ModelRun
from src.database.connection import SyncSessionLocal
from src.utils.logger import logger


async def save_prediction(db: AsyncSession, data: Dict[str, Any]) -> Prediction:
    """Inserts a new prediction record asynchronously."""
    shap_str = json.dumps(data.get("shap_values")) if isinstance(data.get("shap_values"), (dict, list)) else data.get("shap_values")

    prediction_record = Prediction(
        id=data.get("prediction_id") or data.get("id"),
        timestamp=data.get("timestamp", datetime.now(timezone.utc)),
        model_used=data["model_used"],
        ph=data.get("ph"),
        hardness=data.get("Hardness") or data.get("hardness"),
        solids=data.get("Solids") or data.get("solids"),
        chloramines=data.get("Chloramines") or data.get("chloramines"),
        sulfate=data.get("Sulfate") or data.get("sulfate"),
        conductivity=data.get("Conductivity") or data.get("conductivity"),
        organic_carbon=data.get("Organic_carbon") or data.get("organic_carbon"),
        trihalomethanes=data.get("Trihalomethanes") or data.get("trihalomethanes"),
        turbidity=data.get("Turbidity") or data.get("turbidity"),
        prediction=int(data["prediction"]),
        probability=float(data["probability"]),
        confidence=data["confidence"],
        shap_values=shap_str
    )

    db.add(prediction_record)
    await db.commit()
    await db.refresh(prediction_record)
    return prediction_record


async def get_predictions(
    db: AsyncSession,
    limit: int = 10,
    model_filter: Optional[str] = None
) -> List[Prediction]:
    """Retrieves recent prediction records with optional model filter."""
    query = select(Prediction).order_by(Prediction.timestamp.desc()).limit(limit)
    if model_filter:
        query = select(Prediction).where(Prediction.model_used == model_filter).order_by(Prediction.timestamp.desc()).limit(limit)

    result = await db.execute(query)
    return result.scalars().all()


async def save_model_run(db: AsyncSession, data: Dict[str, Any]) -> ModelRun:
    """Inserts a new model benchmark run record asynchronously."""
    run_record = ModelRun(
        model_name=data["model_name"],
        accuracy=data.get("accuracy"),
        precision_score=data.get("precision") or data.get("precision_score"),
        recall=data.get("recall"),
        f1_score=data.get("f1") or data.get("f1_score"),
        roc_auc=data.get("roc_auc"),
        mcc=data.get("mcc"),
        log_loss=data.get("log_loss"),
        mlflow_run_id=data.get("mlflow_run_id")
    )
    db.add(run_record)
    await db.commit()
    await db.refresh(run_record)
    return run_record


async def get_model_runs(db: AsyncSession) -> List[ModelRun]:
    """Retrieves latest benchmark records for all models."""
    query = select(ModelRun).order_by(ModelRun.run_timestamp.desc())
    result = await db.execute(query)
    return result.scalars().all()


def save_model_runs_sync(results_dict: Dict[str, Dict[str, float]]) -> None:
    """Saves multiple model evaluation metrics synchronously."""
    with SyncSessionLocal() as session:
        for model_name, metrics in results_dict.items():
            record = ModelRun(
                model_name=model_name,
                accuracy=metrics.get("accuracy"),
                precision_score=metrics.get("precision"),
                recall=metrics.get("recall"),
                f1_score=metrics.get("f1"),
                roc_auc=metrics.get("roc_auc"),
                mcc=metrics.get("mcc"),
                log_loss=metrics.get("log_loss")
            )
            session.add(record)
        session.commit()
        logger.info(f"Saved {len(results_dict)} model runs to database.")


def get_model_runs_sync() -> List[Dict[str, Any]]:
    """Fetches latest model run metrics synchronously."""
    with SyncSessionLocal() as session:
        records = session.query(ModelRun).order_by(ModelRun.run_timestamp.desc()).all()
        # Keep latest run per model
        seen = set()
        latest = []
        for r in records:
            if r.model_name not in seen:
                seen.add(r.model_name)
                latest.append(r.to_dict())
        return latest
