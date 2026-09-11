---
title: AquaSense AI
emoji: 💧
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
---

# 💧 AquaSense AI

> **Intelligent Water Quality Assessment and Potability Prediction Using Explainable Machine Learning**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Frontend: Vanilla Web](https://img.shields.io/badge/Frontend-Vanilla%20HTML5%20%2F%20CSS3%20%2F%20JS-E34F26.svg?logo=html5&logoColor=white)](https://developer.mozilla.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-EB5424.svg)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8%2B-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 1. Project Overview

Access to safe drinking water is a fundamental human right and a critical public health imperative. Traditional laboratory water quality testing is resource-intensive, slow, and expensive. **AquaSense AI** provides an intelligent, automated, data-driven machine learning system capable of evaluating physicochemical water parameters and instantly predicting potability while explaining its decision-making process with state-of-the-art **Explainable AI (XAI)** techniques.

### 🌟 Key Highlights & Differentiators
- **5 Production ML Classification Models**: Rigorously compares Random Forest (🏆 Champion), XGBoost, Support Vector Machine (SVM), Decision Tree, and Logistic Regression.
- **Tri-Method Explainability (XAI)**: Combines **SHAP** (Shapley Additive exPlanations), **LIME** (Local Interpretable Model-agnostic Explanations), and **Permutation Feature Importance**.
- **Explanation Consistency Analysis**: Evaluates cross-method Spearman rank correlation and classifies feature agreement into *Strong*, *Moderate*, or *Weak* consensus tiers.
- **Probability Calibration & Uncertainty Quantification**: Uses isotonic calibration to provide realistic empirical confidence levels (*Low*, *Moderate*, *High*, *Very High*).
- **Custom Modern Web Dashboard**: Built with pure Vanilla HTML5, modern CSS3 (clinical light theme), and ES6+ JavaScript, directly served by FastAPI on `http://localhost:8000` (single terminal process, no third-party framework overhead).
- **Zero Data Leakage Pipeline**: Custom preprocessing transformers (`GroupedMedianImputer`, `IQRCapper`, `DomainFeatureEngineer`, `RobustScaler`) fitted strictly on training partitions.
- **Audit & Governance**: Persistent SQLite database logging all inferences, execution latencies, and parameters with one-click JSON export.

---

## 🏗️ 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLIENT WEB BROWSER                            │
│           Vanilla HTML5 + Modern CSS3 (Clinical Theme) + JS (Chart.js)  │
│  [🏠 Dashboard] [🔬 Predict] [📊 Data Explorer] [📈 Models] [🧠 XAI]    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ REST HTTP (JSON)
┌────────────────────────────────────▼────────────────────────────────────┐
│                          FASTAPI BACKEND SERVER                         │
│   /predict        /explain        /models        /history      /health  │
└──────────────────┬───────────────────────────────────┬──────────────────┘
                   │                                   │
┌──────────────────▼─────────────────┐       ┌─────────▼──────────────────┐
│         ML & XAI ENGINE            │       │      SQLITE DATABASE       │
│  - WaterQualityPipeline            │       │  - predictions (Audit Log) │
│  - 5 Trained Classifiers           │       │  - model_runs (Telemetry)  │
│  - SHAP Explainer (Global & Local) │       └────────────────────────────┘
│  - LIME Explainer (Surrogates)     │
│  - Permutation Importance          │       ┌────────────────────────────┐
│  - Isotonic Calibrator             │       │    MLFLOW TRACKING SERVER  │
│  - Spearman Consistency Analyzer   │       │  - Metrics & Artifacts     │
└────────────────────────────────────┘       └────────────────────────────┘
```

---

## 📊 3. Dataset & WHO Drinking Water Standards

The system analyzes **3,276 water samples** across 9 physicochemical parameters benchmarked against **World Health Organization (WHO)** safe guidelines:

| Parameter | Unit | Dataset Range | WHO Safe Guideline | Health / Environmental Risk |
|---|---|---|---|---|
| **pH** | pH scale | 0.0 – 14.0 | **6.5 – 8.5** | Corrosivity, heavy metal leaching, gastrointestinal irritation |
| **Hardness** | mg/L | 47.4 – 323.1 | **< 200 – 300 mg/L** | Mineral scaling in distribution infrastructure |
| **Solids (TDS)** | ppm | 320.9 – 61,227.2 | **< 500 – 1,000 ppm** | High mineral/organic mass, unpalatable taste |
| **Chloramines** | ppm | 0.35 – 13.13 | **< 4.0 ppm** | Disinfection byproduct precursor, eye/skin irritation |
| **Sulfate** | mg/L | 129.0 – 481.0 | **< 250 mg/L** | Osmotic laxative effect, bitter mineral taste |
| **Conductivity**| μS/cm | 181.5 – 753.3 | **< 400 μS/cm** | High dissolved ionic charge / industrial effluent |
| **Organic Carbon (TOC)** | ppm | 2.2 – 28.3 | **< 2.0 – 4.0 ppm** | Disinfection byproduct formation, bacterial growth |
| **Trihalomethanes (THMs)** | μg/L | 0.74 – 124.0 | **< 80 μg/L** | Carcinogenic chlorination byproduct |
| **Turbidity** | NTU | 1.45 – 6.74 | **< 5.0 NTU (Ideal < 1.0)** | Pathogen shielding from disinfection |

---

## 🏆 4. Model Performance Leaderboard

Evaluated on the held-out test partition with **Matthews Correlation Coefficient (MCC)** as the primary selection criterion for imbalanced data:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | MCC (Primary) | Status |
|---|---|---|---|---|---|---|---|
| **Random Forest** | **80.0%** | **0.784** | **0.655** | **0.714** | **0.815** | **0.4632** | 🏆 **Champion** |
| **XGBoost** | 78.4% | 0.748 | 0.642 | 0.691 | 0.798 | 0.4410 | Active |
| **SVM (RBF Kernel)** | 75.2% | 0.710 | 0.610 | 0.656 | 0.751 | 0.3890 | Active |
| **Decision Tree** | 70.8% | 0.635 | 0.582 | 0.607 | 0.684 | 0.3120 | Active |
| **Logistic Regression** | 62.4% | 0.528 | 0.498 | 0.513 | 0.612 | 0.1850 | Active |

---

## 🚀 5. Quick Start Guide

### Prerequisites
- Python 3.10+ installed
- Modern Web Browser (Chrome, Edge, Firefox, Safari)

### 1. Installation
```bash
# Clone or navigate to the repository
cd f:/AQUASENSE

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
# Run backend + frontend together
python run_app.py
```

- **Web Dashboard:** Open [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive API Documentation:** Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Run Automated Tests
```bash
pytest tests/ -v
```

---

## 📂 6. Repository Structure

```
AQUASENSE/
├── data/
│   └── raw/water_potability.csv     # 3,276 samples
├── models/                          # Serialized model binaries (.pkl)
│   ├── preprocessor.pkl
│   ├── best_model.pkl               # Random Forest Champion
│   ├── calibrated_best_model.pkl
│   └── [5 model pickles].pkl
├── src/
│   ├── data/                        # Preprocessor, feature engineer, loader
│   ├── models/                      # Trainer, evaluator, calibrator
│   ├── explainability/              # SHAP, LIME, Permutation, Consistency
│   ├── database/                    # SQLAlchemy connection, models, CRUD
│   ├── api/                         # FastAPI main app & routes
│   └── utils/                       # Logger, configuration
├── frontend/                        # Vanilla HTML5 / CSS3 / JS single-page app
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── tests/                           # Pytest test suite (20 tests)
├── Documentation/
│   └── AquaSense_AI_Documentation.md# Comprehensive project documentation
├── run_app.py                       # Single-command launcher
├── requirements.txt                 # Pinned dependencies
└── README.md                        # Project documentation
```

---

## 📄 7. License & Credits
Developed by **Humza** (CodeVibe with Humza).  
Distributed under the **MIT License**.
