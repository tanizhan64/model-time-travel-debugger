# model_time_travel_debugger

import os
import joblib
import pandas as pd
import numpy as np
import shap
import streamlit as st
from streamlit.components.v1 import html
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------
# Configuration
# --------------------------
MODEL_DIR = "models"
DATA_PATHS = {
    "v1": "data/housing_v1.csv",
    "v2": "data/housing_v2.csv"
}
MODEL_PATHS = {
    "v1": f"{MODEL_DIR}/model_v1.pkl",
    "v2": f"{MODEL_DIR}/model_v2.pkl"
}

# --------------------------
# Utility Functions
# --------------------------
def load_data(version):
    return pd.read_csv(DATA_PATHS[version])

def train_model(data, version):
    X = data.drop(columns=["target"])
    y = data["target"]
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATHS[version])
    return model

def load_model(version):
    return joblib.load(MODEL_PATHS[version])

def explain_prediction(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return shap_values, explainer

def get_top_shifted_features(shap_v1, shap_v2, feature_names, top_n=3):
    diff = np.abs(shap_v2 - shap_v1)
    indices = np.argsort(diff[0])[::-1][:top_n]
    return [(feature_names[i], shap_v1[0][i], shap_v2[0][i]) for i in indices]

def generate_explanation(input_row, pred_v1, pred_v2, shifted_features):
    direction = "increased" if pred_v2 > pred_v1 else "decreased"
    change = abs(pred_v2 - pred_v1) / pred_v1 * 100
    lines = [f"🔁 Prediction **{direction}** by {change:.2f}% → from {pred_v1:.2f} to {pred_v2:.2f}."]
    lines.append("🔍 Top changed features:")
    for name, v1, v2 in shifted_features:
        shift = "↑" if abs(v2) > abs(v1) else "↓"
        lines.append(f"- `{name}` importance {shift}: {v1:.3f} → {v2:.3f}")
    return "\n".join(lines)

def evaluate_model(model, X, y):
    preds = model.predict(X)
    return {
        "MAE": mean_absolute_error(y, preds),
        "RMSE": mean_squared_error(y, preds) ** 0.5,  # manually compute sqrt
        "R2": r2_score(y, preds)
    }

def compare_feature_stats(df1, df2):
    drift = []
    for col in df1.columns:
        if col == "target": continue
        mean1, mean2 = df1[col].mean(), df2[col].mean()
        std1, std2 = df1[col].std(), df2[col].std()
        drift.append({
            "feature": col,
            "mean_v1": mean1, "mean_v2": mean2,
            "mean_diff": abs(mean2 - mean1),
            "std_diff": abs(std2 - std1)
        })
    return pd.DataFrame(drift).sort_values(by="mean_diff", ascending=False)

# --------------------------
# Ensure Models Exist
# --------------------------
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

if not os.path.exists(MODEL_PATHS["v1"]) or not os.path.exists(MODEL_PATHS["v2"]):
    df_v1 = load_data("v1")
    df_v2 = load_data("v2")
    train_model(df_v1, "v1")
    train_model(df_v2, "v2")

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Model-Time Travel Debugger", layout="wide")
st.title("🧠 Model-Time Travel: ML Version Debugger")

selected_version = st.selectbox("Select Model Version", ["v1", "v2"])
data = load_data(selected_version)
model = load_model(selected_version)

sample_idx = st.slider("Select Row Index", 0, len(data) - 1, 0)
X = data.drop(columns=["target"])
X_sample = X.iloc[[sample_idx]]

st.write("### 🔍 Input Features")
st.write(X_sample)

pred = model.predict(X_sample)[0]
st.write("### 📈 Model Prediction")
st.success(f"{pred:.2f}")

st.write("### 📊 SHAP Explanation (Bar Chart)")
shap_values, explainer = explain_prediction(model, X_sample)
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
import matplotlib.pyplot as plt
fig = plt.gcf()
st.pyplot(fig)

# --------------------------
# Phase 2: Explain Change
# --------------------------
if st.button("🧠 Explain Change (v1 vs v2)"):
    model_v1 = load_model("v1")
    model_v2 = load_model("v2")

    shap_v1, _ = explain_prediction(model_v1, X_sample)
    shap_v2, _ = explain_prediction(model_v2, X_sample)
    pred_v1 = model_v1.predict(X_sample)[0]
    pred_v2 = model_v2.predict(X_sample)[0]

    shifted = get_top_shifted_features(shap_v1, shap_v2, X_sample.columns)
    explanation = generate_explanation(X_sample.to_dict(orient='records')[0], pred_v1, pred_v2, shifted)

    st.markdown("### 🤖 Local Agent Explanation")
    st.info(explanation)

# --------------------------
# Phase 3: Metrics + Drift
# --------------------------
if st.button("📊 Compare Model Metrics and Drift"):
    df_v1 = load_data("v1")
    df_v2 = load_data("v2")

    model_v1 = load_model("v1")
    model_v2 = load_model("v2")

    X1, y1 = df_v1.drop(columns=["target"]), df_v1["target"]
    X2, y2 = df_v2.drop(columns=["target"]), df_v2["target"]

    metrics_v1 = evaluate_model(model_v1, X1, y1)
    metrics_v2 = evaluate_model(model_v2, X2, y2)

    st.subheader("📈 Model Evaluation")
    st.write("Model v1:", metrics_v1)
    st.write("Model v2:", metrics_v2)

    st.subheader("📉 Feature Drift (mean + std)")
    drift_df = compare_feature_stats(df_v1, df_v2)
    st.dataframe(drift_df)

# --------------------------
# Phase 4: Retraining
# --------------------------
if st.button("🔁 Retrain Both Models"):
    for ver in ["v1", "v2"]:
        df = load_data(ver)
        train_model(df, ver)
    st.success("✅ Models retrained and saved.")
