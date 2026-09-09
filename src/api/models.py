"""Pydantic request and response data contracts for the AquaSense REST API."""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class WaterParameters(BaseModel):
    """Physicochemical water quality input parameters with domain validation."""
    ph: float = Field(..., ge=0.0, le=14.0, description="pH level (acidity/alkalinity)", examples=[7.2])
    Hardness: float = Field(..., ge=0.0, description="Hardness in mg/L (mineral content)", examples=[196.3])
    Solids: float = Field(..., ge=0.0, description="Total dissolved solids in ppm", examples=[22014.0])
    Chloramines: float = Field(..., ge=0.0, description="Chloramines concentration in ppm", examples=[7.12])
    Sulfate: float = Field(..., ge=0.0, description="Sulfate concentration in mg/L", examples=[333.7])
    Conductivity: float = Field(..., ge=0.0, description="Electrical conductivity in μS/cm", examples=[426.2])
    Organic_carbon: float = Field(..., ge=0.0, description="Total organic carbon in ppm", examples=[14.28])
    Trihalomethanes: float = Field(..., ge=0.0, description="Trihalomethanes in μg/L (byproduct)", examples=[66.39])
    Turbidity: float = Field(..., ge=0.0, description="Turbidity in NTU (clarity indicator)", examples=[3.96])


class PredictRequest(WaterParameters):
    """Prediction request payload including model selection."""
    model: str = Field(default="xgboost", description="Model identifier for inference", examples=["xgboost"])


class PredictResponse(BaseModel):
    """Potability prediction output payload."""
    prediction: int = Field(..., description="0 for Not Potable, 1 for Potable")
    label: str = Field(..., description="'Potable' or 'Not Potable'")
    probability_potable: float = Field(..., description="Calibrated probability of water being potable")
    probability_not_potable: float = Field(..., description="Probability of water not being potable")
    confidence: str = Field(..., description="'Low', 'Moderate', 'High', or 'Very High'")
    model_used: str = Field(..., description="Name of the model utilized for prediction")
    prediction_id: str = Field(..., description="Unique UUID identifier for this prediction")


class ExplainRequest(WaterParameters):
    """Payload for requesting local SHAP and LIME explanations."""
    model: str = Field(default="xgboost", description="Model to explain")
    prediction_id: Optional[str] = Field(default=None, description="Optional associated prediction ID")


class ExplainResponse(BaseModel):
    """Local interpretability attribution response."""
    shap_values: Dict[str, float] = Field(..., description="SHAP feature contribution values")
    lime_explanation: Dict[str, float] = Field(..., description="LIME local surrogate feature weights")
    top_factors_for_potable: List[str] = Field(..., description="Features pushing prediction towards potable")
    top_factors_against_potable: List[str] = Field(..., description="Features pushing prediction towards not potable")
    shap_base_value: float = Field(..., description="Expected baseline SHAP value")


class ModelMetricItem(BaseModel):
    """Individual model evaluation summary."""
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    mcc: float
    log_loss: Optional[float] = None
    is_best: bool = False


class ModelsListResponse(BaseModel):
    """Response containing all available models and their evaluation metrics."""
    models: List[ModelMetricItem]
    best_model: str


class FeatureImportanceItem(BaseModel):
    """Feature importance ranking element."""
    feature: str
    importance: float
    rank: int


class FeatureImportanceResponse(BaseModel):
    """Global feature importance response for best model."""
    model_name: str
    method: str = "SHAP"
    rankings: List[FeatureImportanceItem]


class HistoryItem(BaseModel):
    """Past prediction record from database."""
    id: str
    timestamp: str
    model_used: str
    ph: Optional[float]
    hardness: Optional[float]
    solids: Optional[float]
    chloramines: Optional[float]
    sulfate: Optional[float]
    conductivity: Optional[float]
    organic_carbon: Optional[float]
    trihalomethanes: Optional[float]
    turbidity: Optional[float]
    prediction: int
    probability: float
    confidence: str


class HistoryResponse(BaseModel):
    """List of historical prediction records."""
    total: int
    predictions: List[HistoryItem]


class HealthResponse(BaseModel):
    """API health and readiness status."""
    status: str
    app_name: str
    version: str
    models_loaded: int
    models_available: List[str]
    best_model: str
