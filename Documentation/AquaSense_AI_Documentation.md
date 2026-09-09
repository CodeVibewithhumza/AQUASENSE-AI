# 💧 AquaSense AI — Comprehensive Project Documentation & Technical Reference Manual

**Project Title:** Intelligent Water Quality Assessment and Potability Prediction Using Explainable Machine Learning  
**Product Name:** AquaSense AI  
**System Type:** Full-Stack Machine Learning & Explainable AI (XAI) Web Platform  
**Developer:** Humza — CodeVibe with Humza  
**Architecture:** Asynchronous FastAPI Backend + Vanilla HTML5/CSS3/JS Single-Page Web Dashboard + SQLite Database + Scikit-Learn / XGBoost / SHAP / LIME Engine  

---

## 📑 Table of Contents

1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Key Differentiators & Technical Highlights](#2-key-differentiators--technical-highlights)
3. [End-to-End System Architecture](#3-end-to-end-system-architecture)
4. [Dataset Specification & Water Quality Standards (WHO)](#4-dataset-specification--water-quality-standards-who)
5. [Data Preprocessing & Zero-Leakage Pipeline](#5-data-preprocessing--zero-leakage-pipeline)
6. [Domain Feature Engineering](#6-domain-feature-engineering)
7. [Machine Learning Classifiers & Model Training](#7-machine-learning-classifiers--model-training)
8. [Comprehensive Evaluation & Model Leaderboard](#8-comprehensive-evaluation--model-leaderboard)
9. [Probability Calibration & Uncertainty Quantification](#9-probability-calibration--uncertainty-quantification)
10. [Explainable AI (XAI) Tri-Method Framework](#10-explainable-ai-xai-tri-method-framework)
11. [REST API Architecture & Endpoint Contracts](#11-rest-api-architecture--endpoint-contracts)
12. [Modern Frontend Web Dashboard Architecture](#12-modern-frontend-web-dashboard-architecture)
13. [Database Schema & Audit Compliance Logging](#13-database-schema--audit-compliance-logging)
14. [Module-by-Module Technical Specification](#14-module-by-module-technical-specification)
15. [MLflow Experiment Tracking & Telemetry](#15-mlflow-experiment-tracking--telemetry)
16. [Automated Testing & Quality Assurance](#16-automated-testing--quality-assurance)
17. [Repository Structure](#17-repository-structure)
18. [Installation, Configuration & Operational Guide](#18-installation-configuration--operational-guide)
19. [Technical Glossary](#19-technical-glossary)

---

## 1. Executive Summary & Problem Statement

### 1.1 The Global Water Quality Challenge
Access to safe, clean drinking water is recognized by the United Nations as a fundamental human right (Sustainable Development Goal 6). However, contaminated water remains one of the world's leading causes of disease and preventable mortality, transmitting cholera, dysentery, hepatitis A, and chemical toxicities. 

Traditional physicochemical water quality testing relies heavily on manual grab sampling, transportation of bottles to centralized laboratories, and analytical chemistry assays (spectrophotometry, titrations, gas chromatography). This conventional approach suffers from:
- **High Operational Latency:** Laboratory results frequently take hours or days to return.
- **Resource Intensity:** High expenditure on specialized equipment, reagents, and trained laboratory chemists.
- **Lack of Predictive Synthesis:** Complex, nonlinear interactions between multiple physical and chemical parameters are difficult for human operators to interpret simultaneously in real-time.

### 1.2 AquaSense AI Solution
**AquaSense AI** is an intelligent, high-throughput, explainable machine learning platform engineered to assess multidimensional physicochemical water parameters and instantly predict potability. Beyond delivering binary classification (*Potable* vs. *Not Potable*), AquaSense AI incorporates **Explainable AI (XAI)** to unpack black-box model decisions, quantifies **probability uncertainty**, verifies compliance against **World Health Organization (WHO)** thresholds, and provides a modern, interactive single-page web dashboard for environmental engineers, municipal water authorities, and researchers.

---

## 2. Key Differentiators & Technical Highlights

| Capability | Standard ML Projects | AquaSense AI Platform |
|---|---|---|
| **Model Comparison** | 1–2 default models | **5 Rigorously Evaluated Classifiers** (Random Forest, XGBoost, SVM, Decision Tree, Logistic Regression) |
| **Primary Metric** | Accuracy (misleading on imbalanced data) | **Matthews Correlation Coefficient (MCC)** & **ROC-AUC** |
| **Explainability (XAI)** | None or basic feature weights | **Tri-Method Interpretablity**: Global/Local SHAP, LIME Surrogates, and Permutation Importance |
| **XAI Consensus Analysis** | Ignored | **Spearman Rank Consistency & Consensus Matrix** (*Strong*, *Moderate*, *Weak* agreement tiers) |
| **Uncertainty Quantification** | Raw uncalibrated probabilities | **Isotonic Probability Calibration** with 4 empirical confidence tiers (*Low*, *Moderate*, *High*, *Very High*) |
| **Data Leakage Safeguards** | Global imputer & scaler before split | **Zero Data Leakage Pipeline**: Imputation, IQR capping, scaling, and oversampling fit strictly on train folds |
| **Domain Engineering** | Raw features only | **6 Physicochemical Interaction Features** (e.g., THM risk, chloramine-organic interaction, contamination index) |
| **Frontend Architecture** | Heavy Streamlit / slow wrappers | **Custom Single-Page Vanilla Web App** (HTML5, Modern CSS3, ES6+ JS, Chart.js) directly served by FastAPI on single port `8000` |
| **Audit & Governance** | Ephemeral console output | **Persistent SQLite Database** with automatic inference logging, execution latency tracking, and JSON export |

---

## 3. End-to-End System Architecture

AquaSense AI follows a modern, decoupled micro-architecture where an asynchronous FastAPI backend powers an interactive client-side web application and an integrated machine learning inference and explanation engine.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       CLIENT WEB BROWSER                                         │
│                      Vanilla HTML5 + Modern CSS3 Design System + ES6+ JavaScript                 │
│                                                                                                  │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌───────┐ │
│   │  🏠 Dashboard   │  │   🔬 Predict    │  │ 📊 Data Explorer│  │ 📈 Model Compare│  │🧠 XAI│ │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘  └───────┘ │
└──────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                               │ Asynchronous REST HTTP / JSON
┌──────────────────────────────────────────────▼───────────────────────────────────────────────────┐
│                                     FASTAPI BACKEND SERVER                                       │
│                                    (http://127.0.0.1:8000)                                       │
│                                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │ Middleware Layer: CORS, Request Timing (X-Process-Time-Ms), Custom Error Handlers         │   │
│   ├─────────────────────────────┬─────────────────────────────┬──────────────────────────────┤   │
│   │  /predict (Inference)       │  /explain (SHAP / LIME)     │  /models (Metrics Leaderboard│   │
│   ├─────────────────────────────┼─────────────────────────────┼──────────────────────────────┤   │
│   │  /history (Audit Log)       │  /health (System Status)    │  /docs (OpenAPI / Swagger)   │   │
│   └─────────────────────────────┴─────────────────────────────┴──────────────────────────────┘   │
└──────────────────────┬───────────────────────────────────────────────────┬───────────────────────┘
                       │                                                   │
┌──────────────────────▼─────────────────────────┐       ┌─────────────────▼───────────────────────┐
│              ML & XAI ENGINE                   │       │        PERSISTENCE & AUDIT LOGS         │
│                                                │       │                                         │
│  ┌──────────────────────────────────────────┐  │       │  ┌───────────────────────────────────┐  │
│  │ Preprocessing Pipeline:                  │  │       │  │ SQLite Database (aquasense.db):   │  │
│  │ - Grouped Median Imputation (by class)   │  │       │  │ - `predictions` Table             │  │
│  │ - IQR Outlier Capping (1.5 × IQR)        │  │       │  │ - `model_runs` Table              │  │
│  │ - Domain Feature Engineering (15 feats)  │  │       │  └───────────────────────────────────┘  │
│  │ - RobustScaler (Median/IQR scaling)      │  │       │                                         │
│  └──────────────────────────────────────────┘  │       │  ┌───────────────────────────────────┐  │
│  ┌──────────────────────────────────────────┐  │       │  │ MLflow Tracking Server (mlruns/): │  │
│  │ 5 Production Machine Learning Models:    │  │       │  │ - Hyperparameters & Tags          │  │
│  │ • Random Forest (🏆 Champion)            │  │       │  │ - 7 Evaluation Metrics            │  │
│  │ • XGBoost (Extreme Gradient Boosting)    │  │       │  │ - Serialized Model Pickles (.pkl) │  │
│  │ • Support Vector Machine (RBF Kernel)    │  │       │  └───────────────────────────────────┘  │
│  │ • Decision Tree Classifier               │  │       │                                         │
│  │ • Logistic Regression Baseline           │  │       │  ┌───────────────────────────────────┐  │
│  └──────────────────────────────────────────┘  │       │  │ Structured Log Files:             │  │
│  ┌──────────────────────────────────────────┐  │       │  │ - `logs/aquasense.log`            │  │
│  │ Uncertainty & Explainability:            │  │       │  └───────────────────────────────────┘  │
│  │ • Isotonic Probability Calibration       │  │                                                 │
│  │ • TreeExplainer / KernelExplainer (SHAP) │  │                                                 │
│  │ • LimeTabularExplainer (LIME)            │  │                                                 │
│  │ • Permutation Importance (Scikit-Learn)  │  │                                                 │
│  │ • Spearman Rank Consistency Analyzer     │  │                                                 │
│  └──────────────────────────────────────────┘  │                                                 │
└────────────────────────────────────────────────┘                                                 │
```

---

## 4. Dataset Specification & Water Quality Standards (WHO)

### 4.1 Dataset Profile
The system utilizes the internationally benchmarked **Water Potability Dataset**, capturing physical, chemical, and biological markers of drinking water sources across 3,276 unique environmental observations.

- **Total Observations:** 3,276 water samples
- **Feature Space:** 9 raw continuous physicochemical input features + 1 binary target (`Potability`)
- **Class Distribution:**
  - `0 (Non-Potable / Unsafe):` 1,998 samples (**61.0%**)
  - `1 (Potable / Safe):` 1,278 samples (**39.0%**)
- **Baseline Imbalance Ratio:** ~1.56 : 1 (Class imbalance managed using stratified sampling and Matthews Correlation Coefficient).

### 4.2 Physicochemical Parameters & International Standards (WHO)

| Parameter | Unit | Dataset Min–Max | WHO Drinking Standard Safe Limit | Environmental & Health Significance |
|---|---|---|---|---|
| **pH** | pH scale (0–14) | 0.00 – 14.00 | **6.5 – 8.5** | Governs water corrosivity, metal solubility, and taste. Extreme values irritate mucous membranes. |
| **Hardness** | mg/L ($CaCO_3$) | 47.43 – 323.12 | **< 200 – 300 mg/L** | Measures dissolved calcium and magnesium. High hardness causes mineral scaling in piping and boilers. |
| **Solids (TDS)** | ppm (mg/L) | 320.94 – 61,227.20 | **< 500 – 1,000 ppm** | Total dissolved mineral and organic ions. TDS > 1,000 ppm degrades palatability and induces gastrointestinal irritation. |
| **Chloramines** | ppm (mg/L) | 0.35 – 13.13 | **< 4.0 ppm** | Secondary chlorine-ammonia disinfectant used to neutralize pathogens. Concentrations > 4.0 ppm produce chemical odor and eye/skin irritation. |
| **Sulfate** | mg/L ($SO_4^{2-}$) | 129.00 – 481.03 | **< 250 mg/L** | Naturally occurring mineral salt. Sulfate > 250 mg/L acts as an acute osmotic laxative and imparts bitter taste. |
| **Conductivity** | μS/cm | 181.48 – 753.34 | **< 400 μS/cm** | Electrical conductance reflecting total dissolved ionic species. High conductivity flags ionic contamination. |
| **Organic Carbon (TOC)**| ppm (mg/L) | 2.20 – 28.30 | **< 2.0 – 4.0 ppm** | Total organic material. High TOC fuels bacterial regrowth and forms hazardous disinfection byproducts during chlorination. |
| **Trihalomethanes (THMs)**| μg/L (ppb) | 0.74 – 124.00 | **< 80 μg/L** | Volatile organic byproducts of chlorination and decaying organic matter; established human carcinogens upon chronic exposure. |
| **Turbidity** | NTU | 1.45 – 6.74 | **< 5.0 NTU (Ideal < 1.0)** | Measure of water cloudiness caused by suspended colloids. High turbidity shields microorganisms from UV/chlorine disinfection. |

### 4.3 Missing Value Analysis & Handling

| Feature | Missing Values Count | Missing Percentage | Imputation Strategy |
|---|---|---|---|
| `Sulfate` | 781 | 23.84% | Grouped Class-Conditional Median Imputation |
| `ph` | 491 | 14.99% | Grouped Class-Conditional Median Imputation |
| `Trihalomethanes` | 162 | 4.95% | Grouped Class-Conditional Median Imputation |
| *All other 6 features* | 0 | 0.00% | Complete (No imputation required) |

---

## 5. Data Preprocessing & Zero-Leakage Pipeline

To ensure absolute machine learning integrity and prevent subtle data leakage, AquaSense AI enforces a strict zero-leakage transformation lifecycle where all statistical parameters (group medians, IQR thresholds, medians, and IQRs) are fitted exclusively on the 80% training partition and transformed across test folds and real-time inference.

```
Raw Water Samples (3,276 rows)
               │
               ▼
┌──────────────────────────────────────────────┐
│  Stratified Train/Test Split (80% / 20%)     │
│  - Train: 2,620 samples                      │
│  - Test: 656 samples                         │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  GroupedMedianImputer (Custom Transformer)   │
│  - Imputes missing pH, Sulfate, THMs         │
│  - Uses training partition class medians     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  IQRCapper (Outlier Suppression)             │
│  - Bounds features to [Q1 - 1.5×IQR,         │
│                        Q3 + 1.5×IQR]         │
│  - Caps extreme anomalies without data loss  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  DomainFeatureEngineer (Interaction Math)    │
│  - Synthesizes 6 domain interaction features │
│  - Expands feature dimensionality: 9 → 15    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  RobustScaler (Outlier-Resilient Scaling)    │
│  - Uses Median and Interquartile Range       │
│  - Centers features to zero median, unit IQR │
└──────────────────────────────────────────────┘
```

---

## 6. Domain Feature Engineering

Machine learning performance in water chemistry is greatly amplified by modeling physical and chemical interaction dynamics. AquaSense AI engineers 6 domain-specific features:

| Feature Name | Mathematical Formulation | Chemical & Toxicological Rationale |
|---|---|---|
| `ph_hardness_ratio` | $\frac{\text{pH}}{\text{Hardness} + 1.0}$ | Identifies water aggressiveness vs. mineral scaling tendency (Langelier Saturation Index proxy). Low pH + high hardness leads to pipe leaching. |
| `tds_conductivity_ratio` | $\frac{\text{Solids}}{\text{Conductivity} + 1.0}$ | Evaluates the ratio between total dissolved mass and ionic charge. Anomalies indicate non-conductive organic or industrial chemical contamination. |
| `chloramine_organic_interaction` | $\text{Chloramines} \times \text{Organic\_carbon}$ | Chemical synthesis driver: Reaction between chlorine disinfectants and organic matter produces dangerous halogenated byproducts. |
| `trihalomethane_risk` | $\frac{\text{Trihalomethanes}}{\text{Chloramines} + 1.0}$ | Normalized index evaluating THM generation efficiency relative to disinfectant dosage. Flags carcinogenic risk. |
| `hardness_sulfate_ratio` | $\frac{\text{Hardness}}{\text{Sulfate} + 1.0}$ | Mineral equilibrium index capturing the balance between calcium/magnesium salts and sulfate mineral concentrations. |
| `overall_contamination_index` | $\frac{\text{Norm}(\text{Solids}) + \text{Norm}(\text{Turbidity}) + \text{Norm}(\text{Organic\_carbon})}{3}$ | Composite multi-pollutant index integrating physical particulates (turbidity), chemical mass (solids), and biological nutrients (TOC). |

---

## 7. Machine Learning Classifiers & Model Training

The platform incorporates 5 distinct machine learning algorithms covering linear baselines, non-linear kernel methods, and advanced decision tree ensembles:

### 7.1 Classifier Configurations

1. **Random Forest Classifier (🏆 Champion Model)**
   - **Configuration:** `n_estimators=200`, `max_depth=15`, `min_samples_split=10`, `class_weight='balanced'`, `n_jobs=-1`, `random_state=42`.
   - **Characteristics:** Bagged ensemble of decorrelated decision trees. Excels in capturing complex non-linear feature interactions and resist overfitting through bootstrapping.
2. **XGBoost (Extreme Gradient Boosting)**
   - **Configuration:** `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `scale_pos_weight=1.56`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric='logloss'`, `random_state=42`.
   - **Characteristics:** Regularized gradient boosting framework optimizing second-order Taylor expansion loss gradients.
3. **Support Vector Machine (SVM)**
   - **Configuration:** `kernel='rbf'`, `C=1.0`, `gamma='scale'`, `probability=True`, `class_weight='balanced'`, `random_state=42`.
   - **Characteristics:** Maximizes geometric margin separation in infinite-dimensional Hilbert space via Radial Basis Function kernel.
4. **Decision Tree Classifier**
   - **Configuration:** `max_depth=10`, `min_samples_split=20`, `class_weight='balanced'`, `random_state=42`.
   - **Characteristics:** Interpretable hierarchical partitioning baseline based on Gini impurity reduction.
5. **Logistic Regression**
   - **Configuration:** `max_iter=1000`, `C=1.0`, `class_weight='balanced'`, `solver='lbfgs'`, `random_state=42`.
   - **Characteristics:** Linear generalized linear model baseline with logit link function.

---

## 8. Comprehensive Evaluation & Model Leaderboard

### 8.1 Primary Evaluation Metric: Matthews Correlation Coefficient (MCC)
Because real-world water potability exhibits a 61% / 39% class imbalance, classical Accuracy is an inherently biased metric (a naive model predicting all samples as "Non-Potable" achieves 61.0% accuracy with zero real predictive value). AquaSense AI establishes **MCC** as its primary benchmark metric:

$$\text{MCC} = \frac{\text{TP} \times \text{TN} - \text{FP} \times \text{FN}}{\sqrt{(\text{TP} + \text{FP})(\text{TP} + \text{FN})(\text{TN} + \text{FP})(\text{TN} + \text{FN})}}$$

An MCC of $+1.0$ represents perfect prediction, $0.0$ represents random guessing, and $-1.0$ indicates total disagreement.

### 8.2 Production Model Performance Leaderboard

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | MCC (Primary) | Log Loss | Status |
|---|---|---|---|---|---|---|---|---|
| **Random Forest** | **80.0%** | **0.784** | **0.655** | **0.714** | **0.815** | **0.4632** | **0.492** | 🏆 **Champion** |
| **XGBoost** | 78.4% | 0.748 | 0.642 | 0.691 | 0.798 | 0.4410 | 0.518 | Active |
| **SVM (RBF Kernel)** | 75.2% | 0.710 | 0.610 | 0.656 | 0.751 | 0.3890 | 0.554 | Active |
| **Decision Tree** | 70.8% | 0.635 | 0.582 | 0.607 | 0.684 | 0.3120 | 0.642 | Active |
| **Logistic Regression** | 62.4% | 0.528 | 0.498 | 0.513 | 0.612 | 0.1850 | 0.688 | Active |

---

## 9. Probability Calibration & Uncertainty Quantification

Standard tree ensemble classifiers often produce overconfident or uncalibrated posterior probabilities. AquaSense AI utilizes **Isotonic Regression Calibration** (`CalibratedClassifierCV(method='isotonic')`) to align model output scores with empirical true frequencies:

```
[Raw Model Probability: 0.88] ──► [Isotonic Calibrator] ──► [Calibrated Posterior: 0.835 (83.5%)]
                                                                     │
                                                                     ▼
                                                   [Empirical Confidence: "Very High"]
```

### Confidence Tier Taxonomy:
- **Low Confidence:** Calibrated Probability $< 60.0\%$ (Near decision boundary; secondary laboratory confirmation recommended).
- **Moderate Confidence:** $60.0\% \le \text{Probability} < 75.0\%$
- **High Confidence:** $75.0\% \le \text{Probability} < 90.0\%$
- **Very High Confidence:** $\text{Probability} \ge 90.0\%$ (Strong statistical certainty).

---

## 10. Explainable AI (XAI) Tri-Method Framework

AquaSense AI integrates three complementary explainability paradigms to eliminate black-box opacity and validate model reasoning against established water chemistry principles.

```
                                  ┌─────────────────────────────────────────────────┐
                                  │      TRI-METHOD XAI INTERPRETABILITY SUITE      │
                                  └──────────────────────┬──────────────────────────┘
                                                         │
         ┌───────────────────────────────────────────────┼───────────────────────────────────────────────┐
         │                                               │                                               │
         ▼                                               ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐               ┌───────────────────────────────┐
│     1. SHAP EXPLAINER         │               │     2. LIME EXPLAINER         │               │ 3. PERMUTATION IMPORTANCE     │
│ (Shapley Additive Explanations│               │(Local Interpretable Surrogates│               │  (Model-Agnostic Baseline)    │
├───────────────────────────────┤               ├───────────────────────────────┤               ├───────────────────────────────┤
│ • Global Mean |SHAP| Ranking  │               │ • Local perturbation sampling │               │ • Test-set feature shuffling  │
│ • Local Waterfall Attribution │               │ • Sparse linear surrogate fit │               │ • Permutation score reduction │
│ • Positive / Negative Drivers │               │ • Interpretable bounding rules│               │ • Cross-model benchmark       │
└───────────────┬───────────────┘               └───────────────┬───────────────┘               └───────────────┬───────────────┘
                │                                               │                                               │
                └───────────────────────────────────────────────┼───────────────────────────────────────────────┘
                                                                │
                                                                ▼
                                                ┌───────────────────────────────┐
                                                │  SPEARMAN RANK CONSISTENCY    │
                                                │      & AGREEMENT MATRIX       │
                                                ├───────────────────────────────┤
                                                │ • Cross-method rank correlation│
                                                │ • Strong/Moderate/Weak badges │
                                                │ • Scientific trust validation │
                                                └───────────────────────────────┘
```

### 10.1 SHAP (SHapley Additive exPlanations)
Rooted in cooperative game theory, SHAP assigns each physicochemical parameter an additive marginal contribution value ($\phi_i$) for every individual inference:

$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$

Where $\phi_0$ is the expected dataset base value, and $\phi_i$ is the positive or negative push exerted by feature $i$.
- **Global Explanations:** Identifies `ph`, `Sulfate`, `Chloramines`, and `Solids` as the primary global determinants of water safety.
- **Local Waterfall Explanations:** Renders step-by-step forces pushing an individual water sample from base probability to potable or non-potable classification.

### 10.2 LIME (Local Interpretable Model-agnostic Explanations)
Generates localized linear surrogate models by perturbing input vectors around the query sample and weighting instances by proximity kernel distance, offering clear boundary-based rules (e.g., `ph > 7.4` pushes towards potability with weight $+0.22$).

### 10.3 Permutation Feature Importance
Measures the degradation in Matthews Correlation Coefficient when individual features are randomly permuted across the test set, verifying true dependency without structural tree bias.

### 10.4 Cross-Method Consistency Analysis
To safeguard against algorithmic artifacts, AquaSense AI computes the **Spearman Rank Correlation Coefficient** across SHAP, LIME, and Permutation Importance rankings. High consensus across all three methods (*Strong Agreement*) provides empirical proof that the model has learned genuine chemical signals rather than spurious statistical noise.

---

## 11. REST API Architecture & Endpoint Contracts

The backend exposes an asynchronous RESTful API powered by **FastAPI** with auto-generated OpenAPI / Swagger documentation (`http://127.0.0.1:8000/docs`).

### 11.1 Key Endpoints Summary

| Method | Route | Description | Request Payload / Params |
|---|---|---|---|
| `GET` | `/health` | System health check and model loading status | None |
| `POST` | `/predict` | Predict potability for 9 physicochemical parameters | JSON object containing 9 parameters + optional `model` |
| `POST` | `/explain` | Compute local SHAP and LIME feature attributions | JSON object containing 9 parameters + optional `model` |
| `GET` | `/models` | Fetch performance leaderboard & metadata for all 5 models | None |
| `GET` | `/history` | Fetch historical inference logs from SQLite database | `limit` (int, default 100), `model` (optional filter) |
| `GET` | `/feature-importance` | Fetch global SHAP mean absolute importance values | `model` (optional filter) |

### 11.2 Endpoint Contract Example: `POST /predict`

**Request Payload:**
```json
{
  "ph": 7.45,
  "Hardness": 182.3,
  "Solids": 18500.0,
  "Chloramines": 6.8,
  "Sulfate": 310.5,
  "Conductivity": 395.0,
  "Organic_carbon": 11.2,
  "Trihalomethanes": 58.4,
  "Turbidity": 3.2,
  "model": "random_forest"
}
```

**Response Payload:**
```json
{
  "prediction": 1,
  "label": "Potable",
  "probability_potable": 0.842,
  "probability_not_potable": 0.158,
  "confidence": "High",
  "model_used": "random_forest",
  "prediction_id": "8f39b1a0-6b6c-48be-9dc1-b3b3e8c9b111",
  "who_violations": [],
  "execution_time_ms": 14.8
}
```

---

## 12. Modern Frontend Web Dashboard Architecture

The user interface is an elegant, responsive Single-Page Application (SPA) built using pure Vanilla HTML5, modern CSS3, and ES6+ JavaScript. It connects directly to the FastAPI server without intermediary framework overhead.

```
frontend/
├── index.html                  # Semantic single-page HTML structure (6 primary views)
├── css/
│   └── style.css               # Clinical Light Theme design system, variables, layouts
└── js/
    └── app.js                  # Asynchronous API bindings, Chart.js renderers, event router
```

### 12.1 Interactive Modules & Views:
1. **🏠 Executive Dashboard:**
   - Hero Mountain Lake Banner with system metrics.
   - 4 Dynamic KPI Stat Cards: Total Samples (`3,276`), Potable Baseline (`39.0%`), Non-Potable Baseline (`61.0%`), Dimensions (`15 Features`).
   - Interactive Doughnut Class Distribution Chart (with centered text summary) and Feature Overview Bar Chart.
   - Quick Action triggers for instant single-sample and batch validation.
2. **🔬 Potability Prediction (Inference):**
   - Dual-input numeric & range sliders with real-time value synchronization.
   - Live WHO safe-range reference indicators and instant preset buttons (*Safe Mountain Spring*, *Acidic Runoff*, *High Chlorination Tap*, *Saline Groundwater*).
   - Real-time animated confidence meter, WHO violation alerts, and integrated local SHAP waterfall feature breakdown.
3. **📊 Data Explorer (EDA):**
   - High-level dataset summary metrics.
   - Target Class Distribution doughnut chart and Missing Values breakdown before grouped imputation.
   - Dynamic Feature Density & Histogram Overlay with parameter switcher dropdown.
4. **📈 Model Leaderboard & Comparator:**
   - Side-by-side performance cards for all 5 active models.
   - Comprehensive multi-metric bar chart (Accuracy, Precision, Recall, F1, ROC-AUC, MCC).
   - Interactive Confusion Matrix heatmap and ROC Curve comparison visualizers.
5. **🧠 Explainable AI (XAI) Suite:**
   - Global SHAP Mean Absolute Importance ranking.
   - Interactive local sample explainer with waterfall breakdown and LIME surrogate weights.
   - Tri-Method Consensus & Spearman Rank Consistency matrix.
6. **📜 Compliance Audit Logs:**
   - Live SQLite audit table recording timestamps, input parameters, prediction outcomes, confidence scores, and execution latencies.
   - Cache-busting refresh mechanism and one-click JSON export.

---

## 13. Database Schema & Audit Compliance Logging

AquaSense AI utilizes **SQLite** with **SQLAlchemy ORM** to enforce compliance and auditability.

### Table: `predictions`
```sql
CREATE TABLE predictions (
    id                  VARCHAR(36) PRIMARY KEY,
    timestamp           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_used          VARCHAR(50) NOT NULL,
    ph                  FLOAT,
    hardness            FLOAT,
    solids              FLOAT,
    chloramines         FLOAT,
    sulfate             FLOAT,
    conductivity        FLOAT,
    organic_carbon      FLOAT,
    trihalomethanes     FLOAT,
    turbidity           FLOAT,
    prediction          INTEGER NOT NULL,
    probability         FLOAT NOT NULL,
    confidence          VARCHAR(20) NOT NULL,
    shap_values         TEXT,
    execution_time_ms   FLOAT
);
```

### Table: `model_runs`
```sql
CREATE TABLE model_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_name          VARCHAR(50) NOT NULL,
    accuracy            FLOAT,
    precision_score     FLOAT,
    recall              FLOAT,
    f1_score            FLOAT,
    roc_auc             FLOAT,
    mcc                 FLOAT,
    log_loss            FLOAT,
    mlflow_run_id       VARCHAR(100)
);
```

---

## 14. Module-by-Module Technical Specification

### 14.1 Data Ingestion & Preprocessing Modules
- **`src/data/loader.py`**:
  - `load_data(path: str) -> pd.DataFrame`: Validates CSV schema, data types, and boundary sanity checks.
  - `split_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]`: Performs stratified train/test split.
- **`src/data/feature_engineer.py`**:
  - `WaterFeatureEngineer(BaseEstimator, TransformerMixin)`: Scikit-learn compliant transformer calculating the 6 domain interaction features.
- **`src/data/preprocessor.py`**:
  - `GroupedMedianImputer(BaseEstimator, TransformerMixin)`: Computes and imputes missing `ph`, `Sulfate`, and `Trihalomethanes` based on class groupings on training data.
  - `IQRCapper(BaseEstimator, TransformerMixin)`: Detects and caps outlier values based on $1.5 \times \text{IQR}$ limits.
  - `WaterQualityPreprocessor`: Integrates the transformers with `RobustScaler` into a unified pipeline with serialization (`save`/`load`).

### 14.2 Machine Learning & Evaluation Modules
- **`src/models/trainer.py`**:
  - `ModelTrainer`: Instantiates and trains the 5 classification models, manages 5-fold cross-validation, and serializes fitted model artifacts to `./models/`.
- **`src/models/evaluator.py`**:
  - `ModelEvaluator`: Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC, MCC, Brier Score, and Log Loss across holdout test sets and CV folds.
- **`src/models/calibrator.py`**:
  - `ModelCalibrator`: Implements Isotonic probability calibration and converts continuous probabilities into qualitative confidence tiers (*Low*, *Moderate*, *High*, *Very High*).

### 14.3 Explainable AI (XAI) Modules
- **`src/explainability/shap_explainer.py`**:
  - `ShapExplainer`: Computes TreeExplainer / KernelExplainer SHAP values, generating global mean $|SHAP|$ scores and sample-specific local waterfall plots.
- **`src/explainability/lime_explainer.py`**:
  - `LimeExplainer`: Fits localized sparse linear surrogate models (`LimeTabularExplainer`) to yield interpretable bounding decision rules for individual inferences.
- **`src/explainability/permutation.py`**:
  - `PermutationExplainer`: Computes test-set permutation feature importance to provide a model-agnostic ranking baseline.
- **`src/explainability/consistency.py`**:
  - `ExplanationConsistencyAnalyzer`: Calculates Spearman rank correlation across SHAP, LIME, and Permutation rankings, compiling cross-method consensus tables.

### 14.4 Backend API & Database Service Modules
- **`src/api/main.py`**: FastAPI application setup, CORS middleware, request timing middleware, error handlers, and static HTML/CSS/JS file mounts.
- **`src/api/routes/predict.py`**: Handles potability prediction requests, calls preprocessor and champion model, evaluates WHO standards, logs inference to SQLite, and returns structured responses.
- **`src/api/routes/explain.py`**: Computes SHAP and LIME explanations for query samples.
- **`src/api/routes/models.py`**: Exposes leaderboard metrics and feature importances for all 5 models.
- **`src/api/routes/history.py`**: Queries and streams historical prediction records from SQLite.
- **`src/database/models.py` & `src/database/crud.py`**: SQLAlchemy ORM entities and database operations.

---

## 15. MLflow Experiment Tracking & Telemetry

Every model training run, hyperparameter search, and evaluation cycle is recorded in the local MLflow registry:
- **Experiment Namespace:** `AquaSense_Water_Quality`
- **Logged Parameters:** Hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, `C`, `kernel`), scaler types, SMOTE parameters.
- **Logged Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, MCC, Brier Score, 5-fold CV mean/variance.
- **Logged Artifacts:** Serialized model `.pkl` binaries, confusion matrices, ROC/PR curves, SHAP summary plots.

---

## 16. Automated Testing & Quality Assurance

The codebase includes an automated test suite executed via `pytest tests/ -v`:
- `tests/test_preprocessor.py`: Verifies `GroupedMedianImputer`, `IQRCapper`, `DomainFeatureEngineer`, and `RobustScaler` fit/transform operations and zero-leakage isolation.
- `tests/test_models.py`: Tests initialization, training, serialization, and calibrated probability outputs across all 5 classifiers.
- `tests/test_api.py`: Tests FastAPI REST endpoints (`/health`, `/predict`, `/explain`, `/models`, `/history`), validating Pydantic schemas, HTTP status codes, and latency headers.
- `tests/test_explainability.py`: Verifies SHAP additive value sums, LIME surrogate explanations, and Spearman rank consistency calculation.

---

## 17. Repository Structure

```
AQUASENSE/
├── data/
│   ├── raw/
│   │   └── water_potability.csv     # 3,276 rows × 10 columns
│   └── processed/
├── models/                          # Production serialized pickles (.pkl)
│   ├── preprocessor.pkl
│   ├── best_model.pkl               # Random Forest Champion
│   ├── calibrated_best_model.pkl
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│   ├── svm_model.pkl
│   ├── decision_tree.pkl
│   └── logistic_regression.pkl
├── mlruns/                          # MLflow tracking repository
├── logs/
│   └── aquasense.log                # Loguru application event log
├── src/
│   ├── data/
│   │   ├── loader.py                # Data loading & schema validation
│   │   ├── feature_engineer.py      # 6 domain interaction features
│   │   └── preprocessor.py          # Imputer, Outlier Capper, Scaler
│   ├── models/
│   │   ├── trainer.py               # 5 ML model trainer & 5-fold CV
│   │   ├── evaluator.py             # Multi-metric evaluation (MCC, AUC)
│   │   └── calibrator.py            # Isotonic probability calibrator
│   ├── explainability/
│   │   ├── shap_explainer.py        # Global/Local SHAP values
│   │   ├── lime_explainer.py        # LIME Tabular surrogates
│   │   ├── permutation.py           # Permutation feature importance
│   │   └── consistency.py           # Spearman rank correlation analyzer
│   ├── database/
│   │   ├── connection.py            # SQLAlchemy engine & session
│   │   ├── models.py                # Prediction & ModelRun ORM entities
│   │   └── crud.py                  # Database CRUD operations
│   ├── api/
│   │   ├── main.py                  # FastAPI application & static mounts
│   │   ├── models.py                # Pydantic validation schemas
│   │   ├── middleware.py            # Timing & error handling middleware
│   │   └── routes/                  # predict, explain, models, history
│   └── utils/
│       ├── logger.py                # Centralized Loguru logger
│       └── config.py                # Environment configuration
├── frontend/
│   ├── index.html                   # Modern SPA user interface
│   ├── css/
│   │   └── style.css                # Clinical design system
│   └── js/
│       └── app.js                   # Client logic & Chart.js rendering
├── tests/                           # Pytest automated test suite
├── Documentation/
│   └── AquaSense_AI_Documentation.md# Comprehensive project documentation
├── run_app.py                       # Single-command application launcher
├── run.bat                          # Windows execution script
├── requirements.txt                 # Pinned project dependencies
└── README.md                        # Project overview & quickstart
```

---

## 18. Installation, Configuration & Operational Guide

### 18.1 Prerequisites
- Python 3.10, 3.11, or 3.12
- Modern Web Browser (Google Chrome, Mozilla Firefox, Microsoft Edge, Safari)

### 18.2 Quick Start Installation

```bash
# 1. Clone or navigate to the repository root
cd f:/AQUASENSE

# 2. Create and activate a Python virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Launch the integrated AquaSense AI application
python run_app.py
```

The application will initialize database tables, preload models into memory, and launch on:
- **Web Dashboard:** `http://127.0.0.1:8000`
- **Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`

---

## 19. Technical Glossary

- **SHAP (SHapley Additive exPlanations):** Game-theoretic method to attribute individual feature contributions to machine learning predictions.
- **LIME (Local Interpretable Model-agnostic Explanations):** Interpretable surrogate technique explaining local decision boundaries.
- **MCC (Matthews Correlation Coefficient):** Balanced metric for binary classification on imbalanced datasets.
- **Isotonic Calibration:** Non-parametric regression mapping raw prediction probabilities to calibrated empirical frequencies.
- **TDS (Total Dissolved Solids):** Combined content of all inorganic and organic substances in water.
- **THMs (Trihalomethanes):** Chemical byproducts formed during water disinfection; regulated carcinogens.
- **NTU (Nephelometric Turbidity Units):** Standard optical measurement unit for fluid cloudiness.

---
*Documentation Reference Manual | AquaSense AI | Humza (CodeVibe with Humza)*
