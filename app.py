# model_time_travel_debugger: Phase 1 - SHAP-Based Prediction Viewer

import pandas as pd
import numpy as np
import shap
import joblib
import streamlit as st
from streamlit.components.v1 import html
from sklearn.ensemble import RandomForestRegressor
import os

# --------------------------
# Configurations
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

# --------------------------
# Auto-train Models if Not Found
# --------------------------
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

if not os.path.exists(MODEL_PATHS["v1"]) or not os.path.exists(MODEL_PATHS["v2"]):
    df_v1 = load_data("v1")
    df_v2 = load_data("v2")
    train_model(df_v1, "v1")
    train_model(df_v2, "v2")

# --------------------------
# Streamlit App
# --------------------------
st.title("🧠 Model-Time Travel: Phase 1 - SHAP Viewer")

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

shap_values, explainer = explain_prediction(model, X_sample)
shap.initjs()
shap_html = f"<head>{shap.getjs()}</head><body>{shap.force_plot(explainer.expected_value, shap_values, X_sample, matplotlib=False).data}</body>"
html(shap_html, height=300)
