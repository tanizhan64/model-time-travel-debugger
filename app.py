# app.py — Restructured Layout for Streamlit Cloud (Human-Friendly)

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

# -----------------------
# Setup and Paths
# -----------------------
MODEL_DIR = "models"
DATA_DIR = "data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_PATHS = {
    "v1": f"{MODEL_DIR}/model_v1.pkl",
    "v2": f"{MODEL_DIR}/model_v2.pkl"
}
DATA_PATHS = {
    "v1": f"{DATA_DIR}/housing_v1.csv",
    "v2": f"{DATA_DIR}/housing_v2.csv"
}

# -----------------------
# Utility Functions
# -----------------------
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
        "RMSE": mean_squared_error(y, preds) ** 0.5,
        "R2": r2_score(y, preds)
    }

def explain_row(model, X_sample):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    st.subheader("📊 SHAP Waterfall Explanation")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

# -----------------------
# Header
# -----------------------
st.title("🧠 Model-Time Travel Debugger")
st.sidebar.title("🔥 Highly Recommended")
st.sidebar.markdown("""
- Works best with housing demo first
- Upload your own CSV to test SHAP
- Use retrain to reset the v1/v2 models
- Use "Compare Models" to explain prediction shifts
""")

# Load v1/v2 models if missing
for ver in ["v1", "v2"]:
    if not os.path.exists(MODEL_PATHS[ver]):
        df = pd.read_csv(DATA_PATHS[ver])
        train_and_save_model(df, ver)

# -----------------------
# Section 1: Version + Row
# -----------------------
selected_version = st.selectbox("Select Model Version", ["v1", "v2"])
df = pd.read_csv(DATA_PATHS[selected_version])
model = joblib.load(MODEL_PATHS[selected_version])
X = df.drop(columns=["target"])
y = df["target"]

row_idx = st.slider("Select Row Index", 0, len(X)-1, 0)
X_sample = X.iloc[[row_idx]]
st.write("### 🔍 Selected Input Row")
st.dataframe(X_sample)

# -----------------------
# Section 2: Prediction
# -----------------------
st.write("### 📈 Prediction")
pred = model.predict(X_sample)[0]
st.success(f"Prediction: `{pred:.2f}`")

# -----------------------
# Section 3: SHAP Viz
# -----------------------
explain_row(model, X_sample)

# -----------------------
# Section 4: Compare Explanation
# -----------------------
if st.button("🧠 Explain v1 vs v2 Shift"):
    df_v1 = pd.read_csv(DATA_PATHS["v1"])
    df_v2 = pd.read_csv(DATA_PATHS["v2"])
    X1 = df_v1.drop(columns=["target"])
    y1 = df_v1["target"]
    X2 = df_v2.drop(columns=["target"])
    y2 = df_v2["target"]
    model_v1 = joblib.load(MODEL_PATHS["v1"])
    model_v2 = joblib.load(MODEL_PATHS["v2"])
    metrics_v1 = evaluate_model(model_v1, X1, y1)
    metrics_v2 = evaluate_model(model_v2, X2, y2)

    drift_df = pd.DataFrame({
        "Feature": X1.columns,
        "Mean_v1": X1.mean().values,
        "Mean_v2": X2.mean().values,
        "Std_v1": X1.std().values,
        "Std_v2": X2.std().values
    })
    drift_df["ΔMean"] = drift_df["Mean_v2"] - drift_df["Mean_v1"]
    drift_df["ΔStd"] = drift_df["Std_v2"] - drift_df["Std_v1"]

    # -----------------------
    # Section 5: Metrics + Drift
    # -----------------------
    st.subheader("📊 Metric Comparison")
    st.write("Model v1:", metrics_v1)
    st.write("Model v2:", metrics_v2)
    st.subheader("📉 Feature Drift")
    st.dataframe(drift_df)

# -----------------------
# Section 6: Retrain Models
# -----------------------
if st.button("🔁 Retrain v1 & v2 Models"):
    for ver in ["v1", "v2"]:
        df = pd.read_csv(DATA_PATHS[ver])
        train_and_save_model(df, ver)
    st.success("✅ Models retrained!")
