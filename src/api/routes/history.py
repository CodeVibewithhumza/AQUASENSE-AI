"""Prediction audit history query endpoints."""
from fastapi import APIRouter, Depends, Query
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models import HistoryResponse, HistoryItem
from src.database.connection import get_db
from src.database import crud

router = APIRouter(tags=["History"])


@router.get("/history", response_model=HistoryResponse)
async def get_prediction_history(
    limit: int = Query(default=10, ge=1, le=100, description="Max records to retrieve"),
    model: Optional[str] = Query(default=None, description="Optional model filter"),
    db: AsyncSession = Depends(get_db)
):
    """Retrieves recent water potability assessment audit logs from the database."""
    records = await crud.get_predictions(db, limit=limit, model_filter=model)

    items = []
    for r in records:
        items.append(HistoryItem(
            id=str(r.id),
            timestamp=r.timestamp.isoformat() if r.timestamp else "",
            model_used=r.model_used,
            ph=r.ph,
            hardness=r.hardness,
            solids=r.solids,
            chloramines=r.chloramines,
            sulfate=r.sulfate,
            conductivity=r.conductivity,
            organic_carbon=r.organic_carbon,
            trihalomethanes=r.trihalomethanes,
            turbidity=r.turbidity,
            prediction=r.prediction,
            probability=round(r.probability, 4),
            confidence=r.confidence
        ))

    return HistoryResponse(
        total=len(items),
        predictions=items
    )
