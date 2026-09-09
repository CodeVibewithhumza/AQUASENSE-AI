"""Configuration and global constants for AquaSense AI."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base Directory of Project
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment and defaults."""
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "AquaSense AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Database
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR / 'aquasense.db'}"
    SYNC_DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'aquasense.db'}"

    # MLflow
    MLFLOW_TRACKING_URI: str = str(BASE_DIR / "mlruns")
    MLFLOW_EXPERIMENT_NAME: str = "AquaSense_Water_Quality"

    # Paths
    DATA_PATH: str = str(BASE_DIR / "data" / "raw" / "water_potability.csv")
    MODELS_PATH: str = str(BASE_DIR / "models")
    PROCESSED_DATA_PATH: str = str(BASE_DIR / "data" / "processed")

    # ML Settings
    BEST_MODEL: str = "xgboost"
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.20
    CV_FOLDS: int = 5


settings = Settings()

# Ensure critical runtime directories exist
os.makedirs(settings.MODELS_PATH, exist_ok=True)
os.makedirs(settings.PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(settings.MLFLOW_TRACKING_URI, exist_ok=True)
os.makedirs(BASE_DIR / "data" / "raw", exist_ok=True)
os.makedirs(BASE_DIR / "logs", exist_ok=True)

# WHO Water Quality Guidelines & Reference Ranges
WHO_STANDARDS = {
    "ph": {"min": 6.5, "max": 8.5, "default": 7.2, "unit": "pH scale", "name": "pH Level"},
    "Hardness": {"min": 100.0, "max": 300.0, "default": 196.3, "unit": "mg/L", "name": "Hardness"},
    "Solids": {"min": 500.0, "max": 25000.0, "default": 22014.0, "unit": "ppm", "name": "Total Dissolved Solids"},
    "Chloramines": {"min": 4.0, "max": 10.0, "default": 7.12, "unit": "ppm", "name": "Chloramines"},
    "Sulfate": {"min": 250.0, "max": 400.0, "default": 333.7, "unit": "mg/L", "name": "Sulfate"},
    "Conductivity": {"min": 200.0, "max": 600.0, "default": 426.2, "unit": "μS/cm", "name": "Conductivity"},
    "Organic_carbon": {"min": 5.0, "max": 20.0, "default": 14.28, "unit": "ppm", "name": "Organic Carbon"},
    "Trihalomethanes": {"min": 20.0, "max": 80.0, "default": 66.39, "unit": "μg/L", "name": "Trihalomethanes"},
    "Turbidity": {"min": 1.0, "max": 5.0, "default": 3.96, "unit": "NTU", "name": "Turbidity"},
}

PARAMETER_RANGES = {
    "ph": {"min": 0.0, "max": 14.0, "step": 0.1, "default": 7.0},
    "Hardness": {"min": 40.0, "max": 350.0, "step": 1.0, "default": 150.0},
    "Solids": {"min": 300.0, "max": 62000.0, "step": 100.0, "default": 20000.0},
    "Chloramines": {"min": 0.0, "max": 15.0, "step": 0.1, "default": 7.0},
    "Sulfate": {"min": 100.0, "max": 500.0, "step": 1.0, "default": 333.0},
    "Conductivity": {"min": 150.0, "max": 800.0, "step": 5.0, "default": 400.0},
    "Organic_carbon": {"min": 1.0, "max": 30.0, "step": 0.1, "default": 14.0},
    "Trihalomethanes": {"min": 0.0, "max": 130.0, "step": 0.5, "default": 66.0},
    "Turbidity": {"min": 1.0, "max": 7.0, "step": 0.05, "default": 4.0},
}

# Design System Colors
COLORS = {
    "bg_primary": "#0D1B2A",
    "bg_surface": "#1B2A3B",
    "accent": "#00D4FF",
    "danger": "#FF4B4B",
    "safe": "#00C48C",
    "text_primary": "#E8F4FD",
    "text_muted": "#8BA6BB",
    "border": "#2A3F52",
    "card_bg": "#152538",
    "warning": "#FFAA00"
}

FEATURE_NAMES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

ALL_FEATURE_NAMES = FEATURE_NAMES + [
    "ph_hardness_ratio",
    "tds_conductivity_ratio",
    "chloramine_organic_interaction",
    "trihalomethane_risk",
    "hardness_sulfate_ratio",
    "overall_contamination_index"
]
