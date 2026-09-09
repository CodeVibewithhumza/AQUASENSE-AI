"""Potability inference endpoint."""
import uuid
import pandas as pd
from fastapi import APIRouter, Depends, Request, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models import PredictRequest, PredictResponse
from src.database.connection import get_db
from src.database import crud
from src.models.calibrator import ModelCalibrator
from src.utils.logger import logger

router = APIRouter(tags=["Inference"])


@router.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict_water_potability(
    payload: PredictRequest,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Predicts water potability based on 9 physicochemical parameters:
    - Preprocesses input using the fitted production pipeline.
    - Performs inference using the requested or calibrated best model.
    - Derives empirical uncertainty / confidence tier.
    - Records the prediction into the database asynchronously.
    """
    preprocessor = getattr(request.app.state, "preprocessor", None)
    models = getattr(request.app.state, "models", {})
    best_model_name = getattr(request.app.state, "best_model_name", "xgboost")
    calibrated_model = getattr(request.app.state, "calibrated_model", None)

    if preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model preprocessor is not loaded into memory."
        )

    # Resolve selected model
    model_name = payload.model.lower()
    selected_model = None

    if model_name == "calibrated" and calibrated_model is not None:
        selected_model = calibrated_model
        model_name = f"calibrated_{best_model_name}"
    elif model_name in models:
        selected_model = models[model_name]
    elif model_name == "best" and best_model_name in models:
        selected_model = models[best_model_name]
        model_name = best_model_name
    else:
        # Fallback to best model if model name not matched
        if best_model_name in models:
            selected_model = models[best_model_name]
            model_name = best_model_name
        elif models:
            first_key = list(models.keys())[0]
            selected_model = models[first_key]
            model_name = first_key
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No trained classification models available in memory."
            )

    # 1. Prepare raw input DataFrame
    raw_dict = {
        "ph": payload.ph,
        "Hardness": payload.Hardness,
        "Solids": payload.Solids,
        "Chloramines": payload.Chloramines,
        "Sulfate": payload.Sulfate,
        "Conductivity": payload.Conductivity,
        "Organic_carbon": payload.Organic_carbon,
        "Trihalomethanes": payload.Trihalomethanes,
        "Turbidity": payload.Turbidity
    }
    df_raw = pd.DataFrame([raw_dict])

    # 2. Preprocess input features
    try:
        X_processed = preprocessor.transform(df_raw)
    except Exception as e:
        logger.error(f"Error during input preprocessing: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Preprocessing error: {str(e)}"
        )

    # 3. Model Inference
    try:
        pred_class = int(selected_model.predict(X_processed)[0])

        if hasattr(selected_model, "predict_proba"):
            probs = selected_model.predict_proba(X_processed)[0]
            prob_potable = float(probs[1])
            prob_not_potable = float(probs[0])
        else:
            prob_potable = float(pred_class)
            prob_not_potable = 1.0 - prob_potable

    except Exception as e:
        logger.error(f"Inference failure with model '{model_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {str(e)}"
        )

    # 4. Confidence Tier
    confidence = ModelCalibrator.get_confidence_label(prob_potable)
    label = "Potable" if pred_class == 1 else "Not Potable"
    prediction_id = str(uuid.uuid4())

    # 5. Async Database Record
    try:
        db_payload = {
            "prediction_id": prediction_id,
            "model_used": model_name,
            **raw_dict,
            "prediction": pred_class,
            "probability": prob_potable,
            "confidence": confidence,
            "shap_values": None
        }
        await crud.save_prediction(db, db_payload)
    except Exception as e:
        logger.warning(f"Failed to persist prediction to database: {e}")

    return PredictResponse(
        prediction=pred_class,
        label=label,
        probability_potable=round(prob_potable, 4),
        probability_not_potable=round(prob_not_potable, 4),
        confidence=confidence,
        model_used=model_name,
        prediction_id=prediction_id
    )
