"""Integration tests for FastAPI endpoints."""
import pytest
from starlette.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def api_client():
    """Provides a TestClient context manager that triggers FastAPI lifespan events."""
    with TestClient(app) as client:
        yield client


def test_health_endpoint(api_client):
    """Verifies /health endpoint returns 200 and expected metadata."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "AquaSense AI"
    assert "models_loaded" in data


def test_predict_endpoint_valid(api_client):
    """Verifies /predict endpoint with valid input parameters."""
    payload = {
        "ph": 7.2,
        "Hardness": 180.0,
        "Solids": 21000.0,
        "Chloramines": 7.1,
        "Sulfate": 330.0,
        "Conductivity": 420.0,
        "Organic_carbon": 14.0,
        "Trihalomethanes": 65.0,
        "Turbidity": 3.8,
        "model": "random_forest"
    }
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert data["label"] in ["Potable", "Not Potable"]
    assert 0.0 <= data["probability_potable"] <= 1.0
    assert data["confidence"] in ["Low", "Moderate", "High", "Very High"]
    assert "prediction_id" in data


def test_predict_endpoint_validation_error(api_client):
    """Verifies /predict returns 422 for invalid parameter bounds."""
    invalid_payload = {
        "ph": 25.0,  # Invalid: pH > 14
        "Hardness": 180.0,
        "Solids": 21000.0,
        "Chloramines": 7.1,
        "Sulfate": 330.0,
        "Conductivity": 420.0,
        "Organic_carbon": 14.0,
        "Trihalomethanes": 65.0,
        "Turbidity": 3.8,
        "model": "random_forest"
    }
    response = api_client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "ValidationError"


def test_explain_endpoint(api_client):
    """Verifies /explain endpoint returns SHAP and LIME attributions."""
    payload = {
        "ph": 7.2,
        "Hardness": 180.0,
        "Solids": 21000.0,
        "Chloramines": 7.1,
        "Sulfate": 330.0,
        "Conductivity": 420.0,
        "Organic_carbon": 14.0,
        "Trihalomethanes": 65.0,
        "Turbidity": 3.8,
        "model": "random_forest"
    }
    response = api_client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "shap_values" in data
    assert "lime_explanation" in data
    assert "top_factors_for_potable" in data
    assert "top_factors_against_potable" in data


def test_models_and_feature_importance_endpoints(api_client):
    """Verifies /models and /feature-importance endpoints."""
    res_models = api_client.get("/models")
    assert res_models.status_code == 200
    data_models = res_models.json()
    assert "models" in data_models
    assert len(data_models["models"]) > 0

    res_fi = api_client.get("/feature-importance")
    assert res_fi.status_code == 200
    data_fi = res_fi.json()
    assert "rankings" in data_fi


def test_history_endpoint(api_client):
    """Verifies /history endpoint."""
    response = api_client.get("/history?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "total" in data
