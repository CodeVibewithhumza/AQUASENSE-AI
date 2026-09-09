"""FastAPI Application entrypoint serving the Web Application and REST APIs."""
import os
import contextlib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Request, status, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from src.api.models import HealthResponse
from src.api.middleware import (
    RequestTimingMiddleware,
    validation_exception_handler,
    global_exception_handler
)
from src.api.routes import (
    predict_router,
    explain_router,
    models_router,
    history_router
)
from src.database.connection import init_db
from src.database.crud import get_model_runs_sync, save_model_runs_sync
from src.data.loader import DataLoader
from src.data.preprocessor import WaterQualityPreprocessor
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.models.calibrator import ModelCalibrator
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.consistency import ExplanationConsistencyAnalyzer
from src.utils.config import settings, BASE_DIR, WHO_STANDARDS, PARAMETER_RANGES, FEATURE_NAMES
from src.utils.logger import logger

FRONTEND_DIR = BASE_DIR / "frontend"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that preloads the database, preprocessor,
    all 5 ML classification models, calibrated model, and XAI explainers into memory.
    """
    logger.info("Initializing AquaSense AI Backend Services...")
    await init_db()

    # 1. Load or build preprocessor and models
    models_dir = settings.MODELS_PATH
    preprocessor_file = os.path.join(models_dir, "preprocessor.pkl")

    if not os.path.exists(preprocessor_file):
        logger.info("Artifacts missing. Initiating automatic one-time training bootstrap...")
        dl = DataLoader()
        df = dl.load()
        X_train, X_test, y_train, y_test = dl.split(df)

        preprocessor = WaterQualityPreprocessor()
        X_train_proc = preprocessor.fit_transform(X_train, y_train)
        X_test_proc = preprocessor.transform(X_test)
        X_train_bal, y_train_bal = preprocessor.apply_smote(X_train_proc, y_train)
        preprocessor.save(preprocessor_file)

        trainer = ModelTrainer()
        trained_models = trainer.train_all(X_train_bal, y_train_bal, use_mlflow=False)
        trainer.save_all(trained_models)

        evaluator = ModelEvaluator()
        eval_df = evaluator.evaluate_all(trained_models, X_test_proc, y_test)
        best_name = evaluator.get_best_model(eval_df)
        trainer.save_single("best_model", trained_models[best_name])

        calibrator = ModelCalibrator()
        cal_best = calibrator.calibrate(trained_models[best_name], X_train_bal, y_train_bal)
        calibrator.save(cal_best)

        # Sync save metrics to DB
        eval_dict = eval_df.to_dict(orient="index")
        save_model_runs_sync(eval_dict)
    else:
        preprocessor = WaterQualityPreprocessor.load(preprocessor_file)
        trained_models = ModelTrainer.load_all()

    app.state.preprocessor = preprocessor
    app.state.models = trained_models

    # 2. Determine best model & load calibrated model
    calibrated_path = os.path.join(models_dir, "calibrated_best_model.pkl")
    if os.path.exists(calibrated_path):
        app.state.calibrated_model = ModelCalibrator.load(calibrated_path)
    else:
        app.state.calibrated_model = None

    # Load stored metrics
    db_runs = get_model_runs_sync()
    metrics_cache = {r["model_name"]: r for r in db_runs}
    if not metrics_cache:
        dl = DataLoader()
        df = dl.load()
        _, X_test, _, y_test = dl.split(df)
        X_test_proc = preprocessor.transform(X_test)
        evaluator = ModelEvaluator()
        eval_df = evaluator.evaluate_all(trained_models, X_test_proc, y_test)
        metrics_cache = eval_df.to_dict(orient="index")
        save_model_runs_sync(metrics_cache)

    app.state.evaluation_metrics = metrics_cache

    # Pick best model based on MCC
    best_model_name = "random_forest"
    highest_mcc = -1.0
    for name, m in metrics_cache.items():
        if m.get("mcc", -1.0) > highest_mcc:
            highest_mcc = m["mcc"]
            best_model_name = name

    app.state.best_model_name = best_model_name
    best_model = trained_models.get(best_model_name) or (list(trained_models.values())[0] if trained_models else None)

    # 3. Preload Explainers on background sample
    logger.info(f"Initializing XAI Explainers on best model: '{best_model_name}'...")
    try:
        dl = DataLoader()
        df = dl.load()
        X_train, X_test, y_train, y_test = dl.split(df)
        X_train_proc = preprocessor.transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        shap_explainer = SHAPExplainer().fit(best_model, X_train_proc)
        app.state.shap_explainer = shap_explainer

        lime_explainer = LIMEExplainer().fit(X_train_proc, feature_names=preprocessor.feature_names_out_)
        app.state.lime_explainer = lime_explainer

        # Precompute global feature importance
        sample_for_shap = X_test_proc.head(100)
        shap_exp = shap_explainer.global_explanation(sample_for_shap)
        app.state.global_feature_importance = shap_explainer.get_feature_ranking(shap_exp, preprocessor.feature_names_out_)

        # Precompute consistency analysis
        consistency_analyzer = ExplanationConsistencyAnalyzer()
        rankings, scores = consistency_analyzer.compute_all_rankings(
            best_model, X_train_proc, X_test_proc.head(40), y_test.head(40),
            n_lime_samples=8, n_perm_repeats=6
        )
        corr_df = consistency_analyzer.spearman_correlation_matrix(rankings)
        agree_df = consistency_analyzer.agreement_table(rankings)
        report_text = consistency_analyzer.generate_consistency_report(rankings, corr_df, agree_df)

        app.state.consistency_data = {
            "rankings": rankings,
            "correlation_matrix": corr_df.to_dict(),
            "agreement_table": agree_df.to_dict(orient="records"),
            "report": report_text
        }

    except Exception as e:
        logger.warning(f"Background explainer initialization warning: {e}")
        app.state.shap_explainer = None
        app.state.lime_explainer = None
        app.state.global_feature_importance = [(f, 0.0) for f in preprocessor.feature_names_out_]
        app.state.consistency_data = None

    logger.info("AquaSense AI Backend Services initialized successfully!")
    yield
    logger.info("Shutting down AquaSense AI Backend Services...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Intelligent Water Quality Assessment & Potability Prediction API with Explainable AI",
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# Middleware
app.add_middleware(RequestTimingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Exception Handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

# Routers
app.include_router(predict_router)
app.include_router(explain_router)
app.include_router(models_router)
app.include_router(history_router)


@app.get("/data/summary", tags=["Dataset"])
def get_dataset_summary():
    """Returns dataset summary statistics, missing counts, class counts, and distributions for the web client."""
    dl = DataLoader()
    df = dl.load()
    summary = dl.get_summary(df)

    # Descriptive statistics per feature
    stats = {}
    for col in FEATURE_NAMES:
        col_data = df[col].dropna()
        stats[col] = {
            "mean": round(float(col_data.mean()), 2),
            "std": round(float(col_data.std()), 2),
            "min": round(float(col_data.min()), 2),
            "median": round(float(col_data.median()), 2),
            "max": round(float(col_data.max()), 2),
            "q25": round(float(col_data.quantile(0.25)), 2),
            "q75": round(float(col_data.quantile(0.75)), 2),
            "who_min": WHO_STANDARDS[col]["min"],
            "who_max": WHO_STANDARDS[col]["max"],
            "unit": WHO_STANDARDS[col]["unit"],
            "name": WHO_STANDARDS[col]["name"]
        }

    # Correlation matrix
    corr_df = df[FEATURE_NAMES + ["Potability"]].corr().round(3)
    corr_dict = {
        "columns": list(corr_df.columns),
        "values": corr_df.values.tolist()
    }

    # Distribution histograms (potable vs not potable)
    distributions = {}
    for col in FEATURE_NAMES:
        potable_vals = df[df["Potability"] == 1][col].dropna().tolist()
        non_potable_vals = df[df["Potability"] == 0][col].dropna().tolist()
        distributions[col] = {
            "potable": potable_vals[:200],  # sample for fast transmission
            "non_potable": non_potable_vals[:200]
        }

    return {
        "summary": summary,
        "stats": stats,
        "correlation": corr_dict,
        "distributions": distributions,
        "who_standards": WHO_STANDARDS,
        "parameter_ranges": PARAMETER_RANGES
    }


@app.get("/explain/consistency", tags=["Explainability"])
def get_consistency_data(request: Request):
    """Returns cached XAI consistency rankings, correlation matrix, and agreement table."""
    consistency = getattr(request.app.state, "consistency_data", None)
    if consistency is None:
        raise HTTPException(status_code=503, detail="Consistency data not computed yet.")
    return consistency


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK, tags=["System"])
def health_check():
    """System health check and runtime model inventory."""
    models_loaded = getattr(app.state, "models", {})
    best_model_name = getattr(app.state, "best_model_name", "xgboost")
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        models_loaded=len(models_loaded),
        models_available=list(models_loaded.keys()),
        best_model=best_model_name
    )


# Serve Web Application Static Assets
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=FileResponse, tags=["Web App"])
def serve_index():
    """Serves the AquaSense AI Web Application Single Page Interface."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(
            status_code=404,
            content={"message": "Frontend index.html not found"}
        )
    return FileResponse(str(index_file))
