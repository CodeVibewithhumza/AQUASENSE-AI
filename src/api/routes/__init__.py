"""API route handlers package."""
from src.api.routes.predict import router as predict_router
from src.api.routes.explain import router as explain_router
from src.api.routes.models import router as models_router
from src.api.routes.history import router as history_router

__all__ = [
    "predict_router",
    "explain_router",
    "models_router",
    "history_router"
]
