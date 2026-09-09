"""Script to generate all 7 comprehensive Jupyter notebooks for AquaSense AI."""
import json
import os
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(exist_ok=True)


def create_notebook(filename: str, cells: list):
    """Helper to write a valid Jupyter Notebook JSON."""
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10"
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    filepath = NOTEBOOKS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated notebook: {filepath}")


def markdown_cell(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def code_cell(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


# ==============================================================================
# 01_EDA.ipynb
# ==============================================================================
cells_01 = [
    markdown_cell("""# 💧 AquaSense AI — 01: Exploratory Data Analysis (EDA)
**Project:** Intelligent Water Quality Assessment and Potability Prediction Using Explainable Machine Learning  
**Author:** Humza | NexSham Technologies  

### Overview:
In this notebook, we perform a deep exploratory analysis on the **Water Potability Dataset** (3,276 samples across 9 physicochemical parameters and 1 binary potability label).
We explore data types, missing value patterns, class imbalance, distributions, and correlation structure."""),
    code_cell("""import sys
import os
sys.path.append(os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.loader import DataLoader
from src.utils.config import WHO_STANDARDS, COLORS
from src.utils.visualization import set_plot_style

set_plot_style()
print("Libraries imported successfully.")"""),
    markdown_cell("""## 1. Load Dataset & Schema Verification"""),
    code_cell("""dl = DataLoader(data_path="../data/raw/water_potability.csv")
df = dl.load()
summary = dl.get_summary(df)

print(f"Dataset Shape: {df.shape}")
print(f"Total Rows: {summary['rows']}, Total Columns: {summary['columns']}")
df.head(5)"""),
    markdown_cell("""## 2. Statistical Summary & Descriptive Metrics"""),
    code_cell("""df.describe().T"""),
    markdown_cell("""## 3. Missing Value Analysis
Identify attributes with missing observations and calculate missingness percentages."""),
    code_cell("""missing_df = pd.DataFrame({
    'Missing Values': df.isnull().sum(),
    'Percentage (%)': (df.isnull().mean() * 100).round(2)
})
missing_df = missing_df[missing_df['Missing Values'] > 0]
print(missing_df)

plt.figure(figsize=(8, 4))
sns.barplot(x=missing_df['Percentage (%)'], y=missing_df.index, palette='Blues_r')
plt.title("Missing Values Percentage by Feature", fontsize=12, color=COLORS['accent'])
plt.xlabel("Missing Percentage (%)")
plt.tight_layout()
plt.show()"""),
    markdown_cell("""## 4. Target Class Distribution (Imbalance Analysis)
Potability is coded as: `0 = Not Potable`, `1 = Potable`."""),
    code_cell("""counts = df['Potability'].value_counts()
pcts = df['Potability'].value_counts(normalize=True) * 100

print(f"Class 0 (Not Potable): {counts[0]} ({pcts[0]:.2f}%)")
print(f"Class 1 (Potable): {counts[1]} ({pcts[1]:.2f}%)")

fig, ax = plt.subplots(figsize=(6, 5))
ax.pie(counts, labels=['Not Potable (0)', 'Potable (1)'], autopct='%1.1f%%',
       colors=[COLORS['danger'], COLORS['safe']], startangle=140, explode=[0.05, 0])
plt.title("Target Potability Class Distribution", fontsize=12, color=COLORS['accent'])
plt.tight_layout()
plt.show()"""),
    markdown_cell("""## 5. Feature Distributions & Kernel Density Estimation (KDE)
Examine distribution shape across classes for each water quality indicator."""),
    code_cell("""features = [c for c in df.columns if c != 'Potability']
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, feat in enumerate(features):
    ax = axes[i]
    sns.kdeplot(data=df[df['Potability'] == 0][feat], ax=ax, label='Not Potable (0)', color=COLORS['danger'], fill=True, alpha=0.3)
    sns.kdeplot(data=df[df['Potability'] == 1][feat], ax=ax, label='Potable (1)', color=COLORS['accent'], fill=True, alpha=0.3)
    ax.set_title(feat, color=COLORS['accent'], fontsize=11)
    ax.legend()

plt.suptitle("KDE Distribution by Potability Target", fontsize=15, color=COLORS['text_primary'], y=1.02)
plt.tight_layout()
plt.show()"""),
    markdown_cell("""## 6. Correlation Heatmap
Evaluate pairwise Pearson correlations to identify linear multicollinearity."""),
    code_cell("""plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5)
plt.title("Feature Correlation Matrix", fontsize=13, color=COLORS['accent'])
plt.tight_layout()
plt.show()"""),
    markdown_cell("""## 7. Outlier Detection (Box Plots)"""),
    code_cell("""fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feat in enumerate(features):
    ax = axes[i]
    sns.boxplot(y=df[feat], ax=ax, color=COLORS['accent'])
    ax.set_title(feat, color=COLORS['accent'])

plt.suptitle("Physicochemical Outlier Profiling", fontsize=15, color=COLORS['text_primary'], y=1.02)
plt.tight_layout()
plt.show()""")
]
create_notebook("01_EDA.ipynb", cells_01)


# ==============================================================================
# 02_Preprocessing.ipynb
# ==============================================================================
cells_02 = [
    markdown_cell("""# 💧 AquaSense AI — 02: Preprocessing & Feature Engineering
**Project:** Intelligent Water Quality Assessment and Potability Prediction Using Explainable Machine Learning  

### Overview:
In this notebook, we demonstrate:
1. Domain-specific Feature Engineering (6 interaction terms).
2. Class-aware Grouped Median Imputation (without data leakage).
3. IQR Outlier Capping.
4. SMOTE Balancing on the training partition only.
5. Robust feature scaling."""),
    code_cell("""import sys
import os
sys.path.append(os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import DataLoader
from src.data.feature_engineer import WaterFeatureEngineer
from src.data.preprocessor import WaterQualityPreprocessor, GroupedMedianImputer, IQRCapper
from src.utils.visualization import set_plot_style
from src.utils.config import COLORS

set_plot_style()
print("Preprocessing modules loaded.")"""),
    markdown_cell("""## 1. Split Data (Stratified 80/20)"""),
    code_cell("""dl = DataLoader(data_path="../data/raw/water_potability.csv")
df = dl.load()
X_train, X_test, y_train, y_test = dl.split(df, test_size=0.20, random_state=42)
print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")"""),
    markdown_cell("""## 2. Demonstrate Domain Feature Engineering
Construct domain interaction features:
- `ph_hardness_ratio`
- `tds_conductivity_ratio`
- `chloramine_organic_interaction`
- `trihalomethane_risk`
- `hardness_sulfate_ratio`
- `overall_contamination_index`"""),
    code_cell("""fe = WaterFeatureEngineer()
X_train_fe = fe.fit_transform(X_train, y_train)
print(f"Features before FE: {X_train.shape[1]} -> Features after FE: {X_train_fe.shape[1]}")
X_train_fe[['ph_hardness_ratio', 'tds_conductivity_ratio', 'chloramine_organic_interaction', 'overall_contamination_index']].head()"""),
    markdown_cell("""## 3. Demonstrate Grouped Median Imputation
Imputes missing values using target group medians fitted on train partition."""),
    code_cell("""print("Missing values before imputation:")
print(X_train_fe[['ph', 'Sulfate', 'Trihalomethanes']].isnull().sum())

imputer = GroupedMedianImputer()
X_train_imp = imputer.fit_transform(X_train_fe, y_train)

print("\nMissing values after imputation:")
print(X_train_imp[['ph', 'Sulfate', 'Trihalomethanes']].isnull().sum())"""),
    markdown_cell("""## 4. Demonstrate IQR Outlier Capping"""),
    code_cell("""capper = IQRCapper(factor=1.5)
X_train_cap = capper.fit_transform(X_train_imp)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(y=X_train_imp['Solids'], ax=axes[0], color=COLORS['danger'])
axes[0].set_title("Solids Before Capping")

sns.boxplot(y=X_train_cap['Solids'], ax=axes[1], color=COLORS['safe'])
axes[1].set_title("Solids After 1.5x IQR Capping")
plt.tight_layout()
plt.show()"""),
    markdown_cell("""## 5. End-to-End Pipeline & SMOTE Balancing"""),
    code_cell("""preprocessor = WaterQualityPreprocessor()
X_train_proc = preprocessor.fit_transform(X_train, y_train)
X_test_proc = preprocessor.transform(X_test)

print("Class counts before SMOTE:", y_train.value_counts().to_dict())
X_train_bal, y_train_bal = preprocessor.apply_smote(X_train_proc, y_train)
print("Class counts after SMOTE:", y_train_bal.value_counts().to_dict())

preprocessor.save("../models/preprocessor.pkl")
print("Fitted preprocessor serialized successfully.")""")
]
create_notebook("02_Preprocessing.ipynb", cells_02)


# ==============================================================================
# 03_Model_Training.ipynb
# ==============================================================================
cells_03 = [
    markdown_cell("""# 💧 AquaSense AI — 03: Machine Learning Model Training
**Project:** Intelligent Water Quality Assessment and Potability Prediction Using Explainable Machine Learning  

### Overview:
In this notebook, we train all 8 classification models:
1. Logistic Regression (Baseline)
2. Decision Tree
3. Random Forest
4. XGBoost
5. LightGBM
6. Support Vector Machine (SVM)
7. Multi-Layer Perceptron (MLP)
8. Soft Voting Ensemble (RF + XGB + LGBM)

We also perform 5-fold Stratified Cross-Validation and track parameters with MLflow."""),
    code_cell("""import sys
import os
sys.path.append(os.path.abspath('..'))

import pandas as pd
import numpy as np

from src.data.loader import DataLoader
from src.data.preprocessor import WaterQualityPreprocessor
from src.models.trainer import ModelTrainer
from src.utils.logger import logger

print("Training modules loaded.")"""),
    markdown_cell("""## 1. Load Data & Apply Fitted Preprocessor"""),
    code_cell("""dl = DataLoader(data_path="../data/raw/water_potability.csv")
df = dl.load()
X_train, X_test, y_train, y_test = dl.split(df, test_size=0.20, random_state=42)

preprocessor = WaterQualityPreprocessor.load("../models/preprocessor.pkl")
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

X_train_bal, y_train_bal = preprocessor.apply_smote(X_train_proc, y_train)
print(f"Balanced training set: {X_train_bal.shape}, Test set: {X_test_proc.shape}")"""),
    markdown_cell("""## 2. 5-Fold Stratified Cross-Validation Benchmark"""),
    code_cell("""trainer = ModelTrainer(random_state=42)
cv_results = trainer.cross_validate_all(X_train_bal, y_train_bal, cv=5)

df_cv = pd.DataFrame(cv_results).T
df_cv[['cv_accuracy_mean', 'cv_f1_mean', 'cv_roc_auc_mean', 'cv_mcc_mean']].sort_values(by='cv_mcc_mean', ascending=False)"""),
    markdown_cell("""## 3. Train All 8 Models & Serialize Pickles"""),
    code_cell("""trained_models = trainer.train_all(X_train_bal, y_train_bal, use_mlflow=False)
saved_paths = trainer.save_all(trained_models, path="../models")
print("Models saved successfully to disk:")
for k, v in saved_paths.items():
    print(f"  - {k}: {v}")""")
]
create_notebook("03_Model_Training.ipynb", cells_03)


# ==============================================================================
# 04_Evaluation.ipynb
# ==============================================================================
cells_04 = [
    markdown_cell("""# 💧 AquaSense AI — 04: Model Evaluation & Benchmark
**Project:** Intelligent Water Quality Assessment and Potability Prediction Using Explainable Machine Learning  

### Overview:
In this notebook, we evaluate all 8 trained models on the untouched test set across 7 performance metrics:
- Accuracy, Precision, Recall, F1 Score, ROC-AUC, Matthews Correlation Coefficient (MCC), and Log Loss.
We also generate confusion matrices, ROC curves, PR curves, and select the best model."""),
    code_cell("""import sys
import os
sys.path.append(os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data.loader import DataLoader
from src.data.preprocessor import WaterQualityPreprocessor
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.models.calibrator import ModelCalibrator
from src.utils.visualization import set_plot_style

set_plot_style()
print("Evaluation modules loaded.")"""),
    markdown_cell("""## 1. Load Test Data & Models"""),
    code_cell("""dl = DataLoader(data_path="../data/raw/water_potability.csv")
df = dl.load()
_, X_test, _, y_test = dl.split(df, test_size=0.20, random_state=42)

preprocessor = WaterQualityPreprocessor.load("../models/preprocessor.pkl")
X_test_proc = preprocessor.transform(X_test)

models = ModelTrainer.load_all(path="../models")
print(f"Loaded {len(models)} models.")"""),
    markdown_cell("""## 2. Multi-Metric Benchmark Table"""),
    code_cell("""evaluator = ModelEvaluator()
results_df = evaluator.evaluate_all(models, X_test_proc, y_test)
results_df.sort_values(by='mcc', ascending=False)"""),
    markdown_cell("""## 3. Identify & Declare Best Model"""),
    code_cell("""best_model_name = evaluator.get_best_model(results_df, metric='mcc')
print(f"🏆 Best Model Selected by MCC: {best_model_name}")
best_model = models[best_model_name]
ModelTrainer().save_single("best_model", best_model, path="../models")"""),
    markdown_cell("""## 4. Confusion Matrices (2x4 Grid)"""),
    code_cell("""cm_fig = evaluator.plot_confusion_matrices(models, X_test_proc, y_test)
plt.show()"""),
    markdown_cell("""## 5. ROC & Precision-Recall Curves"""),
    code_cell("""roc_fig = evaluator.plot_roc_curves(models, X_test_proc, y_test)
plt.show()

pr_fig = evaluator.plot_pr_curves(models, X_test_proc, y_test)
plt.show()"""),
    markdown_cell("""## 6. Model Comparison Bar Chart"""),
    code_cell("""comp_fig = evaluator.plot_model_comparison(results_df)
plt.show()"""),
    markdown_cell("""## 7. Probability Calibration on Best Model"""),
    code_cell("""calibrator = ModelCalibrator(method='isotonic')
calibrated_model = calibrator.calibrate(best_model, X_test_proc, y_test)
calibrator.save(calibrated_model, path="../models/calibrated_best_model.pkl")

cal_fig = calibrator.plot_calibration_curve(calibrated_model, best_model, X_test_proc, y_test)
plt.show()""")
]
create_notebook("04_Evaluation.ipynb", cells_04)


# ==============================================================================
# 05_SHAP_Analysis.ipynb
# ==============================================================================
cells_05 = [
    markdown_cell("""# 💧 AquaSense AI — 05: SHAP Interpretability Analysis
**Project:** Intelligent Water Quality Assessment and Potability Prediction Using Explainable Machine Learning  

### Overview:
In this notebook, we apply **SHAP (SHapley Additive exPlanations)** to compute game-theoretic feature attributions for our top model.
We analyze:
- Global Beeswarm summary plot
- Global mean absolute SHAP bar chart
- Local Waterfall plots across 3 distinct test cases (High confidence potable, High confidence non-potable, Uncertain/boundary)
- Feature dependence & interaction plots"""),
    code_cell("""import sys
import os
sys.path.append(os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data.loader import DataLoader
from src.data.preprocessor import WaterQualityPreprocessor
from src.models.trainer import ModelTrainer
from src.explainability.shap_explainer import SHAPExplainer
from src.utils.visualization import set_plot_style

set_plot_style()
print("SHAP modules loaded.")"""),
    markdown_cell("""## 1. Load Data, Preprocessor & Best Model"""),
    code_cell("""dl = DataLoader(data_path="../data/raw/water_potability.csv")
df = dl.load()
X_train, X_test, y_train, y_test = dl.split(df, test_size=0.20, random_state=42)

preprocessor = WaterQualityPreprocessor.load("../models/preprocessor.pkl")
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

best_model = ModelTrainer.load_single("best_model", path="../models")
print("Model loaded:", type(best_model))"""),
    markdown_cell("""## 2. Fit SHAP Explainer & Compute Global Values"""),
    code_cell("""shap_explainer = SHAPExplainer()
shap_explainer.fit(best_model, X_train_proc)
shap_explanation = shap_explainer.global_explanation(X_test_proc)
print("SHAP computation complete.")"""),
    markdown_cell("""## 3. Global Beeswarm Summary Plot"""),
    code_cell("""beeswarm_fig = shap_explainer.plot_beeswarm(shap_explanation, X_test_proc, max_display=12)
plt.show()"""),
    markdown_cell("""## 4. Mean Absolute SHAP Importance Bar Chart"""),
    code_cell("""bar_fig = shap_explainer.plot_bar(shap_explanation, max_display=12)
plt.show()

ranking = shap_explainer.get_feature_ranking(shap_explanation, preprocessor.feature_names_out_)
pd.DataFrame(ranking, columns=['Feature', 'Mean |SHAP Value|']).head(10)"""),
    markdown_cell("""## 5. Local Waterfall Plots (3 Representative Test Cases)"""),
    code_cell("""# Case 1: High confidence potable
# Case 2: High confidence not potable
# Case 3: Uncertain / boundary sample
probs = best_model.predict_proba(X_test_proc)[:, 1]

idx_potable = int(np.argmax(probs))
idx_not_potable = int(np.argmin(probs))
idx_uncertain = int(np.argmin(np.abs(probs - 0.5)))

print(f"Sample #{idx_potable} Prob: {probs[idx_potable]:.3f} (High Potable)")
fig1 = shap_explainer.plot_waterfall(shap_explanation, idx=idx_potable)
plt.show()

print(f"Sample #{idx_not_potable} Prob: {probs[idx_not_potable]:.3f} (High Not Potable)")
fig2 = shap_explainer.plot_waterfall(shap_explanation, idx=idx_not_potable)
plt.show()

print(f"Sample #{idx_uncertain} Prob: {probs[idx_uncertain]:.3f} (Uncertain)")
fig3 = shap_explainer.plot_waterfall(shap_explanation, idx=idx_uncertain)
plt.show()"""),
    markdown_cell("""## 6. Feature Dependence Plots"""),
    code_cell("""fig_dep = shap_explainer.plot_dependence(shap_explanation, X_test_proc, feature='Sulfate')
plt.show()""")
]
create_notebook("05_SHAP_Analysis.ipynb", cells_05)


# ==============================================================================
# 06_LIME_Analysis.ipynb
# ==============================================================================
cells_06 = [
    markdown_cell("""# 💧 AquaSense AI — 06: LIME Local Interpretability Analysis
**Project:** Intelligent Water Quality Assessment and Potability Prediction Using Explainable Machine Learning  

### Overview:
In this notebook, we apply **LIME (Local Interpretable Model-agnostic Explanations)** to fit sparse linear surrogate models around individual water quality predictions.
We evaluate:
- Explanations for 5 correctly predicted potable samples
- Explanations for 5 correctly predicted non-potable samples
- Explanations for 5 misclassified samples (error diagnosis)
- Aggregate global LIME feature importance"""),
    code_cell("""import sys
import os
sys.path.append(os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data.loader import DataLoader
from src.data.preprocessor import WaterQualityPreprocessor
from src.models.trainer import ModelTrainer
from src.explainability.lime_explainer import LIMEExplainer
from src.utils.visualization import set_plot_style

set_plot_style()
print("LIME modules loaded.")"""),
    markdown_cell("""## 1. Load Data & Fit LIME Explainer"""),
    code_cell("""dl = DataLoader(data_path="../data/raw/water_potability.csv")
df = dl.load()
X_train, X_test, y_train, y_test = dl.split(df, test_size=0.20, random_state=42)

preprocessor = WaterQualityPreprocessor.load("../models/preprocessor.pkl")
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

best_model = ModelTrainer.load_single("best_model", path="../models")

lime_explainer = LIMEExplainer(random_state=42)
lime_explainer.fit(X_train_proc, feature_names=preprocessor.feature_names_out_)
print("LIME explainer initialized.")"""),
    markdown_cell("""## 2. Explain Single Prediction"""),
    code_cell("""sample_idx = 0
sample_row = X_test_proc.iloc[sample_idx]
exp_plot = lime_explainer.plot_explanation(best_model, sample_row, idx=sample_idx)
plt.show()"""),
    markdown_cell("""## 3. Error Analysis: Explaining Misclassified Samples"""),
    code_cell("""y_pred = best_model.predict(X_test_proc)
misclassified_indices = np.where(y_pred != y_test.values)[0]
print(f"Total misclassified test samples: {len(misclassified_indices)}")

for i in misclassified_indices[:3]:
    row = X_test_proc.iloc[i]
    true_cls = y_test.iloc[i]
    pred_cls = y_pred[i]
    print(f"--- Sample #{i} (True: {true_cls}, Pred: {pred_cls}) ---")
    fig = lime_explainer.plot_explanation(best_model, row, idx=i)
    plt.show()"""),
    markdown_cell("""## 4. Aggregate Global LIME Feature Importance"""),
    code_cell("""lime_ranking = lime_explainer.get_feature_ranking(best_model, X_test_proc, n_samples=25)
df_lime = pd.DataFrame(lime_ranking, columns=['Feature', 'Mean Absolute LIME Weight'])
df_lime.head(10)""")
]
create_notebook("06_LIME_Analysis.ipynb", cells_06)


# ==============================================================================
# 07_Consistency_Analysis.ipynb
# ==============================================================================
cells_07 = [
    markdown_cell("""# 💧 AquaSense AI — 07: Explanation Consistency Analysis
**Project:** Intelligent Water Quality Assessment and Potability Prediction Using Explainable Machine Learning  

### Overview:
In this notebook, we perform the **cross-method XAI consistency analysis** comparing:
1. **SHAP** (Game-theoretic Shapley values)
2. **LIME** (Local sparse linear surrogates)
3. **Permutation Importance** (Empirical metric decay under perturbation)

We compute:
- Spearman rank correlation matrix
- Agreement classification table (Strong / Moderate / Weak)
- Normalized side-by-side attribution charts
- Plain English trustworthiness conclusions"""),
    code_cell("""import sys
import os
sys.path.append(os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data.loader import DataLoader
from src.data.preprocessor import WaterQualityPreprocessor
from src.models.trainer import ModelTrainer
from src.explainability.consistency import ExplanationConsistencyAnalyzer
from src.utils.visualization import set_plot_style

set_plot_style()
print("Consistency analysis modules loaded.")"""),
    markdown_cell("""## 1. Load Data, Preprocessor & Best Model"""),
    code_cell("""dl = DataLoader(data_path="../data/raw/water_potability.csv")
df = dl.load()
X_train, X_test, y_train, y_test = dl.split(df, test_size=0.20, random_state=42)

preprocessor = WaterQualityPreprocessor.load("../models/preprocessor.pkl")
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

best_model = ModelTrainer.load_single("best_model", path="../models")
print("Model loaded successfully.")"""),
    markdown_cell("""## 2. Compute Rankings Across All 3 Interpretability Methods"""),
    code_cell("""analyzer = ExplanationConsistencyAnalyzer(random_state=42)
rankings, scores = analyzer.compute_all_rankings(
    best_model, X_train_proc, X_test_proc.head(50), y_test.head(50),
    n_lime_samples=15, n_perm_repeats=10
)

df_rankings = pd.DataFrame(rankings)
df_rankings.index = range(1, len(df_rankings) + 1)
df_rankings.head(10)"""),
    markdown_cell("""## 3. Spearman Rank Correlation Matrix"""),
    code_cell("""corr_matrix = analyzer.spearman_correlation_matrix(rankings)
print("Spearman Rank Correlation Matrix:")
display(corr_matrix)

heat_fig = analyzer.plot_correlation_heatmap(corr_matrix)
plt.show()"""),
    markdown_cell("""## 4. Feature Agreement Classification Table"""),
    code_cell("""agreement_df = analyzer.agreement_table(rankings)
display(agreement_df)"""),
    markdown_cell("""## 5. Normalized Consistency Comparison Bar Chart"""),
    code_cell("""comp_fig = analyzer.plot_consistency_comparison(rankings, scores)
plt.show()"""),
    markdown_cell("""## 6. Plain English Interpretability & Trustworthiness Report"""),
    code_cell("""report = analyzer.generate_consistency_report(rankings, corr_matrix, agreement_df)
print(report)""")
]
create_notebook("07_Consistency_Analysis.ipynb", cells_07)

print("All 7 Jupyter Notebooks generated successfully!")
