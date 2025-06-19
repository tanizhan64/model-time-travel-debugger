# 🧠 Model-Time Travel Debugger — Phase 2: CSV Upload Support

import pandas as pd
import numpy as np
import shap
import joblib
import streamlit as st
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Model-Time Travel Debugger", layout="wide")

# -------------------------------
# Setup
# -------------------------------
MODEL_DIR = "models"
DATA_DIR = "data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_PATHS = {f"v{i}": f"{MODEL_DIR}/model_v{i}.pkl" for i in [1, 2]}
DATA_PATHS = {f"v{i}": f"{DATA_DIR}/housing_v{i}.csv" for i in [1, 2]}

def train_and_save_model(data, version):
    X = data.drop(columns=["target"])
    y = data["target"]
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATHS[version])
    return model

def evaluate_model(model, X, y):
    preds = model.predict(X)
    return {
        "MAE": mean_absolute_error(y, preds),
        "RMSE": np.sqrt(mean_squared_error(y, preds)),
        "R2": r2_score(y, preds)
    }

def explain_row(model, X_sample):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    st.subheader("📊 SHAP Waterfall Explanation")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

def get_explanation_text(pred_v1, pred_v2, top_features):
    delta = pred_v2 - pred_v1
    direction = "increased" if delta > 0 else "decreased"
    percent = abs(delta) / (abs(pred_v1) + 1e-8) * 100
    explanation = f"🔍 Prediction has **{direction} by {percent:.2f}%**, from `{pred_v1:.2f}` to `{pred_v2:.2f}`.\n\n"
    explanation += "Top changing features:\n"
    for feature, value in top_features:
        arrow = "⬆️" if value > 0 else "⬇️"
        explanation += f"- {arrow} `{feature}` changed SHAP by `{value:.3f}`\n"
    return explanation

# -------------------------------
# Header
# -------------------------------
st.title("🧠 Model-Time Travel Debugger (w/ CSV Upload)")

# -------------------------------
# CSV Upload Section
# -------------------------------
st.sidebar.markdown("## 📤 Upload Your Datasets")
for ver in ["v1", "v2"]:
    uploaded = st.sidebar.file_uploader(f"Upload CSV for {ver.upper()}", type=["csv"], key=ver)
    if uploaded is not None:
        df_new = pd.read_csv(uploaded)
        if "target" not in df_new.columns:
            st.sidebar.error("CSV must include a `target` column.")
        else:
            df_new.to_csv(DATA_PATHS[ver], index=False)
            train_and_save_model(df_new, ver)
            st.sidebar.success(f"✅ v{ver} updated and model retrained.")

# -------------------------------
# Select Model and Row
# -------------------------------
selected_version = st.selectbox("Select Model Version", ["v1", "v2"])
df = pd.read_csv(DATA_PATHS[selected_version])
model = joblib.load(MODEL_PATHS[selected_version])
X = df.drop(columns=["target"])
y = df["target"]

row_idx = st.slider("Select Row Index", 0, len(X)-1, 0)
X_sample = X.iloc[[row_idx]]

st.write("### 🔍 Selected Input Row")
st.dataframe(X_sample)

st.write("### 📈 Prediction")
pred = model.predict(X_sample)[0]
st.success(f"Model {selected_version} predicts: `{pred:.2f}`")

explain_row(model, X_sample)

# -------------------------------
# Compare v1 vs v2
# -------------------------------
if st.button("🧠 Explain v1 vs v2 Shift"):
    model_v1 = joblib.load(MODEL_PATHS["v1"])
    model_v2 = joblib.load(MODEL_PATHS["v2"])
    pred_v1 = model_v1.predict(X_sample)[0]
    pred_v2 = model_v2.predict(X_sample)[0]

    shap_v1 = shap.Explainer(model_v1)(X_sample)
    shap_v2 = shap.Explainer(model_v2)(X_sample)
    diff = shap_v2.values[0] - shap_v1.values[0]
    top_idx = np.argsort(np.abs(diff))[::-1][:3]
    top_features = [(X.columns[i], diff[i]) for i in top_idx]

    st.markdown("### 🗣️ Version Explanation")
    st.info(get_explanation_text(pred_v1, pred_v2, top_features))

# -------------------------------
# Metrics + Drift
# -------------------------------
if st.button("📈 View: Model Metrics & Feature Drift"):
    df_v1 = pd.read_csv(DATA_PATHS["v1"])
    df_v2 = pd.read_csv(DATA_PATHS["v2"])
    X1, y1 = df_v1.drop(columns=["target"]), df_v1["target"]
    X2, y2 = df_v2.drop(columns=["target"]), df_v2["target"]
    model_v1 = joblib.load(MODEL_PATHS["v1"])
    model_v2 = joblib.load(MODEL_PATHS["v2"])

    st.markdown("### 📏 Metrics")
    st.markdown("#### Model v1")
    for k, v in evaluate_model(model_v1, X1, y1).items():
        st.markdown(f"- **{k}**: `{v:.4f}`")
    st.markdown("#### Model v2")
    for k, v in evaluate_model(model_v2, X2, y2).items():
        st.markdown(f"- **{k}**: `{v:.4f}`")

    st.markdown("### 🔄 Feature Drift")
    drift_df = pd.DataFrame({
        "Feature": X1.columns,
        "Mean_v1": X1.mean(),
        "Mean_v2": X2.mean(),
        "ΔMean": X2.mean() - X1.mean()
    })
    st.dataframe(drift_df)

# -------------------------------
# Manual Retrain
# -------------------------------
if st.button("🔁 Manual Retrain Both Models"):
    for ver in ["v1", "v2"]:
        df = pd.read_csv(DATA_PATHS[ver])
        train_and_save_model(df, ver)
    st.success("✅ Models retrained successfully.")
