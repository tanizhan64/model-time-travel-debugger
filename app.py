# 🧠 Model-Time Travel Debugger — With Target Column Selection

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

MODEL_DIR = "models"
EXAMPLE_DIR = "data"
UPLOAD_DIR = "user_data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXAMPLE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATHS = {f"v{i}": f"{MODEL_DIR}/model_v{i}.pkl" for i in [1, 2]}
EXAMPLE_PATHS = {f"v{i}": f"{EXAMPLE_DIR}/housing_v{i}.csv" for i in [1, 2]}
UPLOAD_PATHS = {f"v{i}": f"{UPLOAD_DIR}/upload_v{i}.csv" for i in [1, 2]}
TARGET_META = {f"v{i}": f"{UPLOAD_DIR}/target_v{i}.txt" for i in [1, 2]}

def train_and_save_model(data, version, target_col):
    X = data.drop(columns=[target_col])
    y = data[target_col]
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
st.title("🧠 Model-Time Travel Debugger (Target Column Selectable)")
data_mode = st.radio("📦 Choose Data Mode", ["📘 Use Example Dataset", "📁 Upload Your Own CSVs"])

# -------------------------------
# Upload Mode: Select target column
# -------------------------------
if data_mode == "📁 Upload Your Own CSVs":
    for ver in ["v1", "v2"]:
        uploaded = st.sidebar.file_uploader(f"Upload CSV for {ver.upper()}", type=["csv"], key=ver)
        if uploaded:
            df = pd.read_csv(uploaded)
            st.sidebar.markdown(f"**Select Target Column for {ver.upper()}**")
            target_col = st.sidebar.selectbox(f"Target for {ver.upper()}", df.columns, key=f"target_{ver}")
            if target_col:
                df.to_csv(UPLOAD_PATHS[ver], index=False)
                with open(TARGET_META[ver], "w") as f:
                    f.write(target_col)
                train_and_save_model(df, ver, target_col)
                st.sidebar.success(f"✅ {ver.upper()} trained with target: `{target_col}`")

# -------------------------------
# Load Data
# -------------------------------
def get_data(ver):
    if data_mode == "📁 Upload Your Own CSVs" and os.path.exists(UPLOAD_PATHS[ver]):
        return pd.read_csv(UPLOAD_PATHS[ver])
    return pd.read_csv(EXAMPLE_PATHS[ver])

def get_target_col(ver, df):
    if data_mode == "📁 Upload Your Own CSVs" and os.path.exists(TARGET_META[ver]):
        return open(TARGET_META[ver]).read().strip()
    return "target"

# Block if upload mode but no files yet
if data_mode == "📁 Upload Your Own CSVs" and not all(os.path.exists(UPLOAD_PATHS[v]) and os.path.exists(TARGET_META[v]) for v in ["v1", "v2"]):
    st.warning("Please upload CSVs for both v1 and v2 and select a target column.")
    st.stop()

selected_version = st.selectbox("Select Model Version", ["v1", "v2"])
df = get_data(selected_version)
target_col = get_target_col(selected_version, df)
model = joblib.load(MODEL_PATHS[selected_version])
X = df.drop(columns=[target_col])
y = df[target_col]

row_idx = st.slider("Pick Row Index", 0, len(X) - 1, 0)
X_sample = X.iloc[[row_idx]]

st.write("### 🔍 Input Features")
st.dataframe(X_sample)

st.write("### 📈 Prediction")
pred = model.predict(X_sample)[0]
st.success(f"Model {selected_version} predicts: `{pred:.2f}`")

explain_row(model, X_sample)

# -------------------------------
# Explain Shift
# -------------------------------
if st.button("🧠 Explain v1 vs v2 Shift"):
    model_v1 = joblib.load(MODEL_PATHS["v1"])
    model_v2 = joblib.load(MODEL_PATHS["v2"])
    df_v1 = get_data("v1")
    df_v2 = get_data("v2")
    t1 = get_target_col("v1", df_v1)
    t2 = get_target_col("v2", df_v2)
    X1 = df_v1.drop(columns=[t1])
    X2 = df_v2.drop(columns=[t2])
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
if st.button("📈 View Metrics + Drift"):
    df_v1 = get_data("v1")
    df_v2 = get_data("v2")
    t1 = get_target_col("v1", df_v1)
    t2 = get_target_col("v2", df_v2)
    model_v1 = joblib.load(MODEL_PATHS["v1"])
    model_v2 = joblib.load(MODEL_PATHS["v2"])
    X1, y1 = df_v1.drop(columns=[t1]), df_v1[t1]
    X2, y2 = df_v2.drop(columns=[t2]), df_v2[t2]

    st.markdown("### 📏 Evaluation Metrics")
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
if st.button("🔁 Retrain Models"):
    for ver in ["v1", "v2"]:
        df = get_data(ver)
        target = get_target_col(ver, df)
        train_and_save_model(df, ver, target)
    st.success("✅ Models retrained.")
