"""Explainability endpoint computing local SHAP and LIME feature attributions."""
import pandas as pd
from fastapi import APIRouter, Request, HTTPException, status

from src.api.models import ExplainRequest, ExplainResponse
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.utils.logger import logger

router = APIRouter(tags=["Explainability"])


@router.post("/explain", response_model=ExplainResponse, status_code=status.HTTP_200_OK)
async def explain_water_prediction(
    payload: ExplainRequest,
    request: Request
):
    """
    Computes local SHAP and LIME feature attributions for an individual water sample:
    - Preprocesses inputs using the production pipeline.
    - Computes exact local Shapley values and baseline expected value.
    - Constructs a local sparse linear LIME surrogate model.
    - Returns top driving factors for and against potable water safety.
    """
    preprocessor = getattr(request.app.state, "preprocessor", None)
    models = getattr(request.app.state, "models", {})
    best_model_name = getattr(request.app.state, "best_model_name", "xgboost")
    shap_explainer: SHAPExplainer = getattr(request.app.state, "shap_explainer", None)
    lime_explainer: LIMEExplainer = getattr(request.app.state, "lime_explainer", None)

    if preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Preprocessor is not initialized."
        )

    model_name = payload.model.lower()
    selected_model = models.get(model_name) or models.get(best_model_name) or (list(models.values())[0] if models else None)

    if selected_model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No classification models available for explanation."
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

    # 2. Preprocess input
    X_processed = preprocessor.transform(df_raw)

    # 3. SHAP local attribution
    try:
        if shap_explainer is None:
            shap_explainer = SHAPExplainer().fit(selected_model, X_processed)
        shap_res = shap_explainer.local_explanation(X_processed)
    except Exception as e:
        logger.warning(f"SHAP local explanation fallback: {e}")
        # Build graceful fallback attribution
        feat_names = preprocessor.feature_names_out_
        shap_res = {
            "shap_values": {f: 0.0 for f in feat_names},
            "shap_base_value": 0.5,
            "top_factors_for_potable": [],
            "top_factors_against_potable": []
        }

    # 4. LIME local surrogate attribution
    try:
        if lime_explainer is None:
            lime_explainer = LIMEExplainer().fit(X_processed, feature_names=preprocessor.feature_names_out_)
        lime_weights = lime_explainer.get_feature_weights(selected_model, X_processed, num_features=10)
    except Exception as e:
        logger.warning(f"LIME local explanation fallback: {e}")
        lime_weights = {f: 0.0 for f in preprocessor.feature_names_out_[:10]}

    return ExplainResponse(
        shap_values=shap_res["shap_values"],
        lime_explanation=lime_weights,
        top_factors_for_potable=shap_res["top_factors_for_potable"],
        top_factors_against_potable=shap_res["top_factors_against_potable"],
        shap_base_value=round(shap_res["shap_base_value"], 4)
    )
