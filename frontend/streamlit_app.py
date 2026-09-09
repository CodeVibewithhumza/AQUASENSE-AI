"""AquaSense AI — Production Streamlit Dashboard with Dark Navy Scientific Design."""
import os
import requests
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

from src.utils.config import (
    settings,
    COLORS,
    WHO_STANDARDS,
    PARAMETER_RANGES,
    FEATURE_NAMES,
    ALL_FEATURE_NAMES
)
from src.utils.visualization import set_plot_style, create_radar_chart, plot_correlation_matrix_plotly
from src.data.loader import DataLoader
from src.data.preprocessor import WaterQualityPreprocessor
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.models.calibrator import ModelCalibrator
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.consistency import ExplanationConsistencyAnalyzer

# Page configuration
st.set_page_config(
    page_title="AquaSense AI — Water Quality Assessment",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def apply_custom_css():
    """Applies high-contrast dark navy scientific design tokens and custom typography."""
    css = f"""
    <style>
        /* Import Modern Sans-Serif & Monospace Typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@600;700&display=swap');

        /* Global Theme Overrides */
        .stApp {{
            background-color: {COLORS['bg_primary']};
            color: {COLORS['text_primary']};
            font-family: 'Inter', sans-serif;
        }}

        /* Headings */
        h1, h2, h3, h4 {{
            font-family: 'Space Grotesk', sans-serif !important;
            color: {COLORS['accent']} !important;
            letter-spacing: -0.02em;
        }}

        /* Top Header Bar */
        .header-container {{
            background: linear-gradient(135deg, {COLORS['bg_surface']} 0%, #102A43 100%);
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            padding: 24px 28px;
            margin-bottom: 24px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .header-title {{
            font-size: 2.2rem;
            font-weight: 700;
            color: {COLORS['accent']};
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .header-subtitle {{
            color: {COLORS['text_muted']};
            font-size: 0.95rem;
            margin-top: 6px;
        }}

        .badge-live {{
            background-color: rgba(0, 212, 255, 0.15);
            color: {COLORS['accent']};
            border: 1px solid {COLORS['accent']};
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        /* Metric & Stat Cards */
        .stat-card {{
            background-color: {COLORS['bg_surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 16px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-2px);
            border-color: {COLORS['accent']};
        }}
        .stat-label {{
            color: {COLORS['text_muted']};
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }}
        .stat-value {{
            font-size: 1.6rem;
            font-weight: 700;
            color: {COLORS['text_primary']};
            font-family: 'JetBrains Mono', monospace;
        }}

        /* Result Cards */
        .potable-box {{
            background: linear-gradient(135deg, rgba(0, 196, 140, 0.12) 0%, rgba(0, 196, 140, 0.04) 100%);
            border: 2px solid {COLORS['safe']};
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 8px 24px rgba(0, 196, 140, 0.2);
            margin-bottom: 20px;
        }}
        .not-potable-box {{
            background: linear-gradient(135deg, rgba(255, 75, 75, 0.12) 0%, rgba(255, 75, 75, 0.04) 100%);
            border: 2px solid {COLORS['danger']};
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 8px 24px rgba(255, 75, 75, 0.2);
            margin-bottom: 20px;
        }}
        .status-title {{
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }}

        /* Factor Attribution Cards */
        .factor-card {{
            background-color: {COLORS['bg_surface']};
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-left: 4px solid;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
        }}
        .factor-card.positive {{
            border-left-color: {COLORS['safe']};
            background: rgba(0, 196, 140, 0.08);
        }}
        .factor-card.negative {{
            border-left-color: {COLORS['danger']};
            background: rgba(255, 75, 75, 0.08);
        }}

        /* Custom Streamlit Buttons */
        .stButton>button {{
            background: linear-gradient(135deg, {COLORS['accent']} 0%, #0099CC 100%);
            color: #05131E !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 12px 24px !important;
            width: 100% !important;
            transition: all 0.2s ease !important;
        }}
        .stButton>button:hover {{
            box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4) !important;
            transform: translateY(-1px) !important;
        }}

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: {COLORS['bg_surface']};
            padding: 6px;
            border-radius: 10px;
            border: 1px solid {COLORS['border']};
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {COLORS['text_muted']} !important;
            border-radius: 6px !important;
            padding: 8px 20px !important;
            font-weight: 600 !important;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: rgba(0, 212, 255, 0.15) !important;
            color: {COLORS['accent']} !important;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


@st.cache_data
def load_dataset_cached():
    """Loads and caches the raw water quality dataset."""
    dl = DataLoader()
    df = dl.load()
    return df


@st.cache_resource
def load_ml_pipeline():
    """Loads all models and preprocessors from disk."""
    try:
        preprocessor = WaterQualityPreprocessor.load()
        models = ModelTrainer.load_all()
        calibrated = ModelCalibrator.load()
        return preprocessor, models, calibrated
    except Exception as e:
        return None, None, None


def call_api(endpoint: str, method: str = "GET", payload: dict = None):
    """Makes a request to the FastAPI backend with graceful local fallback."""
    url = f"{API_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        if method.upper() == "POST":
            response = requests.post(url, json=payload, timeout=8)
        else:
            response = requests.get(url, params=payload, timeout=8)

        if response.status_code == 200:
            return response.json(), True
        return response.json(), False
    except Exception:
        return None, False


# Apply CSS
apply_custom_css()

# Header
st.markdown(f"""
<div class="header-container">
    <div>
        <div class="header-title">💧 AquaSense AI</div>
        <div class="header-subtitle">Intelligent Water Quality Assessment & Potability Prediction Using Explainable AI</div>
    </div>
    <div class="badge-live">
        <span style="height: 8px; width: 8px; background-color: {COLORS['accent']}; border-radius: 50%; display: inline-block;"></span>
        Production System v1.0
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Explorer",
    "🔬 Predict",
    "📈 Model Compare",
    "🧠 Explain"
])

df_raw = load_dataset_cached()
preprocessor_loaded, models_loaded, calibrated_loaded = load_ml_pipeline()

# ==============================================================================
# TAB 1: DATA EXPLORER
# ==============================================================================
with tab1:
    st.subheader("Dataset Overview & Physicochemical Profiling")
    st.markdown("Exploratory Data Analysis of the 3,276 water quality observations and 9 primary physicochemical water parameters.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Observations</div>
            <div class="stat-value">{len(df_raw):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Features</div>
            <div class="stat-value">9 Raw + 6 Engineered</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        potable_cnt = int(df_raw['Potability'].sum())
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Potable Samples</div>
            <div class="stat-value" style="color: {COLORS['safe']};">{potable_cnt:,} ({potable_cnt/len(df_raw)*100:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        non_potable_cnt = len(df_raw) - potable_cnt
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Non-Potable Samples</div>
            <div class="stat-value" style="color: {COLORS['danger']};">{non_potable_cnt:,} ({non_potable_cnt/len(df_raw)*100:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    # Class Distribution & Missing Values Visualizations
    col_dist, col_missing = st.columns([1, 1])

    with col_dist:
        st.markdown("#### Class Distribution (Imbalanced Target)")
        dist_fig = go.Figure(data=[go.Pie(
            labels=['Not Potable (0)', 'Potable (1)'],
            values=[non_potable_cnt, potable_cnt],
            hole=0.55,
            marker=dict(colors=[COLORS['danger'], COLORS['safe']]),
            textinfo='label+percent',
            textfont=dict(color=COLORS['text_primary'], size=13)
        )])
        dist_fig.update_layout(
            paper_bgcolor=COLORS['bg_surface'],
            plot_bgcolor=COLORS['bg_surface'],
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=30),
            height=320
        )
        st.plotly_chart(dist_fig, use_container_width=True)

    with col_missing:
        st.markdown("#### Missing Values Before Grouped Imputation")
        missing_s = df_raw.isnull().sum()
        missing_df = pd.DataFrame({"Feature": missing_s.index, "Missing Count": missing_s.values})
        missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(by="Missing Count", ascending=True)

        missing_fig = px.bar(
            missing_df,
            x="Missing Count",
            y="Feature",
            orientation="h",
            color="Missing Count",
            color_continuous_scale=[[0.0, COLORS['accent']], [1.0, COLORS['danger']]],
            text="Missing Count"
        )
        missing_fig.update_layout(
            paper_bgcolor=COLORS['bg_surface'],
            plot_bgcolor=COLORS['bg_surface'],
            font=dict(color=COLORS['text_primary']),
            height=320,
            margin=dict(l=20, r=20, t=30, b=30),
            coloraxis_showscale=False
        )
        st.plotly_chart(missing_fig, use_container_width=True)

    # Statistical Summary Table
    st.markdown("#### Parameter Descriptive Statistics")
    st.dataframe(
        df_raw.describe().T.style.background_gradient(cmap="Blues", subset=["mean", "std"]),
        use_container_width=True
    )

    # Feature Distribution Overlays (KDE)
    st.markdown("#### Physicochemical Distribution by Potability Class")
    selected_feature = st.selectbox("Select Feature for Distribution Inspection:", FEATURE_NAMES)

    kde_fig = go.Figure()
    pot_0 = df_raw[df_raw['Potability'] == 0][selected_feature].dropna()
    pot_1 = df_raw[df_raw['Potability'] == 1][selected_feature].dropna()

    kde_fig.add_trace(go.Histogram(
        x=pot_0,
        name='Not Potable (0)',
        opacity=0.6,
        marker_color=COLORS['danger'],
        histnorm='probability density'
    ))
    kde_fig.add_trace(go.Histogram(
        x=pot_1,
        name='Potable (1)',
        opacity=0.6,
        marker_color=COLORS['accent'],
        histnorm='probability density'
    ))
    kde_fig.update_layout(
        barmode='overlay',
        paper_bgcolor=COLORS['bg_surface'],
        plot_bgcolor=COLORS['bg_surface'],
        font=dict(color=COLORS['text_primary']),
        xaxis=dict(title=selected_feature, gridcolor=COLORS['border']),
        yaxis=dict(title="Density", gridcolor=COLORS['border']),
        margin=dict(l=20, r=20, t=30, b=30),
        height=350,
        legend=dict(x=0.8, y=0.95)
    )
    st.plotly_chart(kde_fig, use_container_width=True)

    # Correlation Heatmap
    st.markdown("#### Feature Correlation Heatmap")
    corr_matrix = df_raw.corr()
    corr_fig = plot_correlation_matrix_plotly(corr_matrix)
    st.plotly_chart(corr_fig, use_container_width=True)


# ==============================================================================
# TAB 2: PREDICT
# ==============================================================================
with tab2:
    st.subheader("Water Potability Diagnostic & Inference Engine")
    st.markdown("Input laboratory water test parameters to assess safety according to WHO drinking water quality thresholds.")

    col_input, col_output = st.columns([1.1, 1.0], gap="large")

    with col_input:
        st.markdown("### ⚙️ Parameter Configuration")

        # Preset Buttons
        st.markdown("**Quick Preset Profiles:**")
        preset_cols = st.columns(3)
        if "preset_key" not in st.session_state:
            st.session_state.preset_key = "default"

        if preset_cols[0].button("🛡️ WHO Standard Safe"):
            st.session_state.ph = 7.2
            st.session_state.Hardness = 160.0
            st.session_state.Solids = 18000.0
            st.session_state.Chloramines = 7.1
            st.session_state.Sulfate = 330.0
            st.session_state.Conductivity = 410.0
            st.session_state.Organic_carbon = 13.5
            st.session_state.Trihalomethanes = 60.0
            st.session_state.Turbidity = 3.2

        if preset_cols[1].button("⚠️ High Mineral"):
            st.session_state.ph = 8.8
            st.session_state.Hardness = 280.0
            st.session_state.Solids = 38000.0
            st.session_state.Chloramines = 4.5
            st.session_state.Sulfate = 420.0
            st.session_state.Conductivity = 650.0
            st.session_state.Organic_carbon = 18.2
            st.session_state.Trihalomethanes = 95.0
            st.session_state.Turbidity = 5.8

        if preset_cols[2].button("☣️ Contaminated"):
            st.session_state.ph = 4.2
            st.session_state.Hardness = 85.0
            st.session_state.Solids = 46000.0
            st.session_state.Chloramines = 11.5
            st.session_state.Sulfate = 180.0
            st.session_state.Conductivity = 720.0
            st.session_state.Organic_carbon = 24.0
            st.session_state.Trihalomethanes = 110.0
            st.session_state.Turbidity = 6.4

        model_choice = st.selectbox(
            "Select Machine Learning Model:",
            [
                "random_forest",
                "xgboost",
                "lightgbm",
                "voting_ensemble",
                "svm",
                "mlp",
                "decision_tree",
                "logistic_regression"
            ],
            format_func=lambda x: f"{x.replace('_', ' ').title()} (Recommended)" if x in ["random_forest", "voting_ensemble"] else x.replace('_', ' ').title()
        )

        st.markdown("---")

        # Sliders with realistic bounds
        in_ph = st.slider("pH Level (WHO: 6.5 – 8.5):", min_value=0.0, max_value=14.0, value=st.session_state.get('ph', 7.0), step=0.1)
        in_hardness = st.slider("Hardness (mg/L):", min_value=40.0, max_value=350.0, value=st.session_state.get('Hardness', 150.0), step=1.0)
        in_solids = st.slider("Total Dissolved Solids (ppm):", min_value=300.0, max_value=62000.0, value=st.session_state.get('Solids', 20000.0), step=100.0)
        in_chloramines = st.slider("Chloramines (ppm):", min_value=0.0, max_value=15.0, value=st.session_state.get('Chloramines', 7.0), step=0.1)
        in_sulfate = st.slider("Sulfate (mg/L):", min_value=100.0, max_value=500.0, value=st.session_state.get('Sulfate', 333.0), step=1.0)
        in_cond = st.slider("Conductivity (μS/cm):", min_value=150.0, max_value=800.0, value=st.session_state.get('Conductivity', 400.0), step=5.0)
        in_organic = st.slider("Organic Carbon (ppm):", min_value=1.0, max_value=30.0, value=st.session_state.get('Organic_carbon', 14.0), step=0.1)
        in_trihalo = st.slider("Trihalomethanes (μg/L):", min_value=0.0, max_value=130.0, value=st.session_state.get('Trihalomethanes', 66.0), step=0.5)
        in_turb = st.slider("Turbidity (NTU):", min_value=1.0, max_value=7.0, value=st.session_state.get('Turbidity', 4.0), step=0.05)

        analyze_btn = st.button("🔬 Analyze Water Quality")

    with col_output:
        st.markdown("### 📊 Assessment Report")

        if analyze_btn:
            payload = {
                "ph": float(in_ph),
                "Hardness": float(in_hardness),
                "Solids": float(in_solids),
                "Chloramines": float(in_chloramines),
                "Sulfate": float(in_sulfate),
                "Conductivity": float(in_cond),
                "Organic_carbon": float(in_organic),
                "Trihalomethanes": float(in_trihalo),
                "Turbidity": float(in_turb),
                "model": model_choice
            }

            with st.spinner("Executing neural inference & computing XAI Shapley values..."):
                # Call FastAPI or fallback
                api_res, ok = call_api("predict", "POST", payload)

                if ok and api_res:
                    pred_class = api_res["prediction"]
                    prob_potable = api_res["probability_potable"]
                    prob_not_potable = api_res["probability_not_potable"]
                    conf = api_res["confidence"]
                    model_used = api_res["model_used"]
                    pid = api_res["prediction_id"]
                else:
                    # Fallback local inference
                    raw_df = pd.DataFrame([payload])[FEATURE_NAMES]
                    X_proc = preprocessor_loaded.transform(raw_df)
                    m = models_loaded.get(model_choice) or list(models_loaded.values())[0]
                    pred_class = int(m.predict(X_proc)[0])
                    probs = m.predict_proba(X_proc)[0]
                    prob_potable = float(probs[1])
                    prob_not_potable = float(probs[0])
                    conf = ModelCalibrator.get_confidence_label(prob_potable)
                    model_used = model_choice
                    pid = "local-uuid"

                # Explanation call
                exp_res, exp_ok = call_api("explain", "POST", payload)
                if not (exp_ok and exp_res):
                    # Fallback local explanation
                    raw_df = pd.DataFrame([payload])[FEATURE_NAMES]
                    X_proc = preprocessor_loaded.transform(raw_df)
                    shap_exp = SHAPExplainer().fit(models_loaded.get(model_choice, list(models_loaded.values())[0]), X_proc)
                    exp_res = shap_exp.local_explanation(X_proc)

            # Display Status Box
            if pred_class == 1:
                st.markdown(f"""
                <div class="potable-box">
                    <div class="status-title" style="color: {COLORS['safe']};">✅ POTABLE WATER</div>
                    <div style="color: {COLORS['text_primary']}; font-size: 1.1rem; font-weight: 600;">Safe for Human Consumption</div>
                    <div style="color: {COLORS['text_muted']}; font-size: 0.85rem; margin-top: 4px;">Confidence Level: <strong style="color: {COLORS['safe']};">{conf.upper()}</strong></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="not-potable-box">
                    <div class="status-title" style="color: {COLORS['danger']};">❌ NOT POTABLE</div>
                    <div style="color: {COLORS['text_primary']}; font-size: 1.1rem; font-weight: 600;">Unsafe for Direct Consumption</div>
                    <div style="color: {COLORS['text_muted']}; font-size: 0.85rem; margin-top: 4px;">Confidence Level: <strong style="color: {COLORS['danger']};">{conf.upper()}</strong></div>
                </div>
                """, unsafe_allow_html=True)

            # Confidence Gauge Indicator
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_potable * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Potability Probability", 'font': {'color': COLORS['text_primary'], 'size': 14}},
                number={'suffix': "%", 'font': {'color': COLORS['accent'], 'size': 24, 'family': 'JetBrains Mono'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLORS['text_muted']},
                    'bar': {'color': COLORS['safe'] if prob_potable >= 0.5 else COLORS['danger']},
                    'bgcolor': COLORS['bg_surface'],
                    'borderwidth': 1,
                    'bordercolor': COLORS['border'],
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(255, 75, 75, 0.15)'},
                        {'range': [50, 100], 'color': 'rgba(0, 196, 140, 0.15)'}
                    ],
                    'threshold': {
                        'line': {'color': COLORS['accent'], 'width': 3},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            gauge_fig.update_layout(
                paper_bgcolor=COLORS['bg_surface'],
                font=dict(color=COLORS['text_primary']),
                height=220,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Explainability Factors breakdown
            st.markdown("#### 🧠 Primary Predictive Drivers (SHAP Local Attribution)")

            top_for = exp_res.get("top_factors_for_potable", [])
            top_against = exp_res.get("top_factors_against_potable", [])
            shap_dict = exp_res.get("shap_values", {})

            col_pos, col_neg = st.columns(2)

            with col_pos:
                st.markdown("<strong style='color: #00C48C;'>Top Factors FOR Potability:</strong>", unsafe_allow_html=True)
                if top_for:
                    for f in top_for[:3]:
                        val = shap_dict.get(f, 0.0)
                        st.markdown(f"""
                        <div class="factor-card positive">
                            <span>{f}</span>
                            <span style="color: {COLORS['safe']}; font-weight: 700;">+{val:.3f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No strong positive drivers.")

            with col_neg:
                st.markdown("<strong style='color: #FF4B4B;'>Top Factors AGAINST Potability:</strong>", unsafe_allow_html=True)
                if top_against:
                    for f in top_against[:3]:
                        val = shap_dict.get(f, 0.0)
                        st.markdown(f"""
                        <div class="factor-card negative">
                            <span>{f}</span>
                            <span style="color: {COLORS['danger']}; font-weight: 700;">{val:.3f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No strong negative drivers.")

        else:
            st.info("👈 Adjust parameters on the left and click **'Analyze Water Quality'** to run the complete diagnostic.")


# ==============================================================================
# TAB 3: MODEL COMPARE
# ==============================================================================
with tab3:
    st.subheader("Model Benchmark & Multi-Metric Evaluation")
    st.markdown("Comprehensive comparison of all 8 machine learning models evaluated on the test set across 7 standard and robust metrics.")

    # Call /models endpoint or fallback to offline evaluator
    models_res, ok = call_api("models")
    if ok and models_res:
        models_data = models_res["models"]
        eval_df = pd.DataFrame(models_data)
        eval_df = eval_df.set_index("name")
        best_model_name = models_res.get("best_model", "random_forest")
    else:
        # Fallback local computation
        dl = DataLoader()
        _, X_test, _, y_test = dl.split(df_raw)
        X_test_proc = preprocessor_loaded.transform(X_test)
        ev = ModelEvaluator()
        eval_df = ev.evaluate_all(models_loaded, X_test_proc, y_test)
        best_model_name = ev.get_best_model(eval_df)

    # Highlight best model badge
    st.markdown(f"""
    <div style="background: rgba(0, 212, 255, 0.1); border: 1px solid {COLORS['accent']}; border-radius: 8px; padding: 14px 20px; margin-bottom: 20px;">
        <span style="font-size: 1.1rem; font-weight: 700; color: {COLORS['accent']};">⭐ Selected Best Model: {best_model_name.replace('_', ' ').title()}</span>
        <span style="color: {COLORS['text_muted']}; margin-left: 12px;">(Top Matthews Correlation Coefficient on imbalanced validation set)</span>
    </div>
    """, unsafe_allow_html=True)

    # Full Metrics Table
    st.markdown("#### All Models Performance Benchmark")
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "mcc", "log_loss"]
    available_cols = [c for c in metric_cols if c in eval_df.columns]

    st.dataframe(
        eval_df[available_cols].style.highlight_max(axis=0, color="#004D40").format("{:.4f}"),
        use_container_width=True
    )

    col_roc, col_radar = st.columns([1.1, 0.9])

    with col_roc:
        st.markdown("#### Grouped Metric Comparison")
        bar_fig = px.bar(
            eval_df.reset_index(),
            x="name",
            y=["accuracy", "f1", "roc_auc", "mcc"],
            barmode="group",
            color_discrete_sequence=[COLORS['accent'], COLORS['safe'], '#FFAA00', '#9D4EDD']
        )
        bar_fig.update_layout(
            paper_bgcolor=COLORS['bg_surface'],
            plot_bgcolor=COLORS['bg_surface'],
            font=dict(color=COLORS['text_primary']),
            xaxis=dict(title="Classifier", tickangle=-30),
            yaxis=dict(title="Score", range=[0, 1.0]),
            margin=dict(l=20, r=20, t=30, b=30),
            height=380,
            legend=dict(orientation="h", y=1.15, x=0.2)
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    with col_radar:
        st.markdown("#### Top-3 Models Radar Profile")
        radar_fig = create_radar_chart(eval_df)
        st.plotly_chart(radar_fig, use_container_width=True)


# ==============================================================================
# TAB 4: EXPLAIN
# ==============================================================================
with tab4:
    st.subheader("Explainable AI (XAI) & Consistency Analysis")
    st.markdown("Evaluating feature attribution robustness across **SHAP**, **LIME**, and **Permutation Importance** to verify model decision consistency.")

    xai_sub_tab1, xai_sub_tab2 = st.tabs(["🌐 Global Explanations & Consistency", "🔍 Local Instance Explainer"])

    with xai_sub_tab1:
        st.markdown("### 1. Global Interpretability Analysis")

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("#### Global Feature Ranking (SHAP)")
            feat_res, feat_ok = call_api("feature-importance")
            if feat_ok and feat_res:
                rankings_list = feat_res["rankings"]
                df_feat_imp = pd.DataFrame(rankings_list)
            else:
                dl = DataLoader()
                X_train, X_test, _, _ = dl.split(df_raw)
                X_test_proc = preprocessor_loaded.transform(X_test)
                best_m = models_loaded.get("random_forest", list(models_loaded.values())[0])
                shap_exp = SHAPExplainer().fit(best_m, preprocessor_loaded.transform(X_train))
                shap_vals = shap_exp.global_explanation(X_test_proc.head(100))
                ranking = shap_exp.get_feature_ranking(shap_vals, preprocessor_loaded.feature_names_out_)
                df_feat_imp = pd.DataFrame([{"feature": f, "importance": s, "rank": i+1} for i, (f, s) in enumerate(ranking)])

            feat_bar_fig = px.bar(
                df_feat_imp.head(10).sort_values(by="importance", ascending=True),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=[[0.0, COLORS['bg_surface']], [1.0, COLORS['accent']]],
                text_auto=".3f"
            )
            feat_bar_fig.update_layout(
                paper_bgcolor=COLORS['bg_surface'],
                plot_bgcolor=COLORS['bg_surface'],
                font=dict(color=COLORS['text_primary']),
                height=380,
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(feat_bar_fig, use_container_width=True)

        with col_g2:
            st.markdown("#### SHAP Summary Impact Distribution")
            with st.spinner("Generating SHAP beeswarm summary..."):
                dl = DataLoader()
                X_train, X_test, _, _ = dl.split(df_raw)
                X_train_proc = preprocessor_loaded.transform(X_train)
                X_test_proc = preprocessor_loaded.transform(X_test)
                best_m = models_loaded.get("random_forest", list(models_loaded.values())[0])

                shap_exp = SHAPExplainer().fit(best_m, X_train_proc)
                shap_vals = shap_exp.global_explanation(X_test_proc.head(80))

                beeswarm_fig = shap_exp.plot_beeswarm(shap_vals, X_test_proc.head(80), max_display=8)
                st.pyplot(beeswarm_fig)
                plt.close(beeswarm_fig)

        st.markdown("---")
        st.markdown("### 2. Explanation Consistency Analysis (SHAP vs LIME vs Permutation)")

        with st.spinner("Computing pairwise rank correlation & agreement matrix..."):
            analyzer = ExplanationConsistencyAnalyzer()
            dl = DataLoader()
            X_train, X_test, y_train, y_test = dl.split(df_raw)
            X_train_proc = preprocessor_loaded.transform(X_train)
            X_test_proc = preprocessor_loaded.transform(X_test)
            best_m = models_loaded.get("random_forest", list(models_loaded.values())[0])

            rankings, scores = analyzer.compute_all_rankings(
                best_m, X_train_proc, X_test_proc.head(40), y_test.head(40),
                n_lime_samples=8, n_perm_repeats=6
            )
            corr_df = analyzer.spearman_correlation_matrix(rankings)
            agree_df = analyzer.agreement_table(rankings)
            report_text = analyzer.generate_consistency_report(rankings, corr_df, agree_df)

        col_c1, col_c2 = st.columns([1.1, 0.9])

        with col_c1:
            st.markdown("#### Method Agreement Classification Table")
            def color_agreement(val):
                if val == "Strong":
                    return f"color: {COLORS['safe']}; font-weight: bold;"
                elif val == "Moderate":
                    return "color: #FFAA00; font-weight: bold;"
                else:
                    return f"color: {COLORS['danger']}; font-weight: bold;"

            # Cross-version pandas Styler compatibility (map in pandas >= 2.1 / 3.0, applymap in older)
            agree_styler = agree_df.style
            if hasattr(agree_styler, "map"):
                styled_agree = agree_styler.map(color_agreement, subset=["Agreement"])
            else:
                styled_agree = agree_styler.applymap(color_agreement, subset=["Agreement"])

            st.dataframe(
                styled_agree,
                use_container_width=True
            )

        with col_c2:
            st.markdown("#### Spearman Rank Correlation Matrix")
            corr_heat = plot_correlation_matrix_plotly(corr_df)
            st.plotly_chart(corr_heat, use_container_width=True)

        st.info(report_text)

    with xai_sub_tab2:
        st.markdown("### Individual Sample Attribution Inspection")
        sample_idx = st.slider("Select Test Sample Index:", min_value=0, max_value=min(50, len(X_test_proc)-1), value=2)

        test_row = X_test_proc.iloc[[sample_idx]]
        true_label = "Potable" if y_test.iloc[sample_idx] == 1 else "Not Potable"

        st.markdown(f"**Sample #{sample_idx} Ground Truth:** `{true_label}`")

        col_loc_shap, col_loc_lime = st.columns(2)

        with col_loc_shap:
            st.markdown("#### SHAP Local Attribution")
            shap_fig = shap_exp.plot_waterfall(shap_vals, idx=min(sample_idx, len(shap_vals)-1), max_display=8)
            st.pyplot(shap_fig)
            plt.close(shap_fig)

        with col_loc_lime:
            st.markdown("#### LIME Local Attribution")
            lime_obj = LIMEExplainer().fit(X_train_proc, feature_names=preprocessor_loaded.feature_names_out_)
            lime_fig = lime_obj.plot_explanation(best_m, test_row, idx=sample_idx, num_features=8)
            st.pyplot(lime_fig)
            plt.close(lime_fig)
