"""SQLAlchemy ORM models for inference predictions and model evaluation benchmarks."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime, Text
from src.database.connection import Base


class Prediction(Base):
    """Stores individual water potability inference records with XAI local attributions."""
    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    model_used = Column(String(50), nullable=False, index=True)

    # Raw water physicochemical parameters
    ph = Column(Float, nullable=True)
    hardness = Column(Float, nullable=True)
    solids = Column(Float, nullable=True)
    chloramines = Column(Float, nullable=True)
    sulfate = Column(Float, nullable=True)
    conductivity = Column(Float, nullable=True)
    organic_carbon = Column(Float, nullable=True)
    trihalomethanes = Column(Float, nullable=True)
    turbidity = Column(Float, nullable=True)

    # Prediction outcomes
    prediction = Column(Integer, nullable=False)        # 0 = Not Potable, 1 = Potable
    probability = Column(Float, nullable=False)         # Probability of Potable (class 1)
    confidence = Column(String(20), nullable=False)      # Low / Moderate / High / Very High
    shap_values = Column(Text, nullable=True)           # Serialized JSON string of SHAP feature attributions

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "model_used": self.model_used,
            "ph": self.ph,
            "hardness": self.hardness,
            "solids": self.solids,
            "chloramines": self.chloramines,
            "sulfate": self.sulfate,
            "conductivity": self.conductivity,
            "organic_carbon": self.organic_carbon,
            "trihalomethanes": self.trihalomethanes,
            "turbidity": self.turbidity,
            "prediction": self.prediction,
            "probability": self.probability,
            "confidence": self.confidence,
            "shap_values": self.shap_values
        }


class ModelRun(Base):
    """Stores benchmark evaluation metrics for model versions and training experiments."""
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    model_name = Column(String(50), nullable=False, index=True)

    # Primary performance metrics
    accuracy = Column(Float, nullable=True)
    precision_score = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    mcc = Column(Float, nullable=True)
    log_loss = Column(Float, nullable=True)

    mlflow_run_id = Column(String(100), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "run_timestamp": self.run_timestamp.isoformat() if self.run_timestamp else None,
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "precision_score": self.precision_score,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "mcc": self.mcc,
            "log_loss": self.log_loss,
            "mlflow_run_id": self.mlflow_run_id
        }
