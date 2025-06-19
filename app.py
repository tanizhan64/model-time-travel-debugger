# app.py — Final Unified App with Upload Support and SHAP Fix

import pandas as pd
import numpy as np
import shap
import joblib
import streamlit as st
import os
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Model-Time Travel Debugger", layout="wide")

# --- Paths ---
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

# --- Utils ---
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
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    force_plot = shap.force_plot(
        explainer.expected_value, shap_values, X_sample, matplotlib=False
    )
    components.html(force_plot.html(), height=300)

# -------------------------
# 🔘 Data Source Selector
# -------------------------
st.title("🧠 Model-Time Travel Debugger")

source = st.radio("📊 Choose Data Source", ["Use Example Housing Dataset", "Upload Your Own CSV"])

# -------------------------
# 📁 Mode 1: Example Dataset
# -------------------------
if source == "Use Example Housing Dataset":
    st.header("🏠 Using Example Housing Dataset")

    # Load or train models
    for ver in ["v1", "v2"]:
        if not os.path.exists(MODEL_PATHS[ver]):
            df = pd.read_csv(DATA_PATHS[ver])
            train_and_save_model(df, ver)

    selected_version = st.selectbox("Select Model Version", ["v1", "v2"])
    df = pd.read_csv(DATA_PATHS[selected_version])
    model = joblib.load(MODEL_PATHS[selected_version])
    X = df.drop(columns=["target"])
    y = df["target"]

    row_idx = st.slider("Select Row Index", 0, len(df)-1, 0)
    X_sample = X.iloc[[row_idx]]
    st.write("### 🔍 Selected Input Row")
    st.dataframe(X_sample)

    st.write("### 📈 Prediction")
    pred = model.predict(X_sample)[0]
    st.success(f"Prediction: `{pred:.2f}`")

    st.write("### 📊 SHAP Explanation")
    explain_row(model, X_sample)

    if st.button("🔬 Compare Metrics & Drift"):
        df_v1 = pd.read_csv(DATA_PATHS["v1"])
        df_v2 = pd.read_csv(DATA_PATHS["v2"])
        X1, y1 = df_v1.drop(columns=["target"]), df_v1["target"]
        X2, y2 = df_v2.drop(columns=["target"]), df_v2["target"]
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

        st.subheader("📊 Metric Comparison")
        st.write("Model v1:", metrics_v1)
        st.write("Model v2:", metrics_v2)
        st.subheader("📉 Feature Drift")
        st.dataframe(drift_df)

# -------------------------
# 📤 Mode 2: Upload Dataset
# -------------------------
else:
    st.header("📤 Upload Your CSV and Retrain")
    file = st.file_uploader("Upload a CSV file", type=["csv"])
    if file:
        df_user = pd.read_csv(file)
        st.write("### Data Preview")
        st.dataframe(df_user.head())

        with st.form("user_train_form"):
            target_col = st.selectbox("🎯 Choose Target Column", df_user.columns)
            submitted = st.form_submit_button("Train and Explain")

        if submitted:
            X = df_user.drop(columns=[target_col])
            y = df_user[target_col]
            model = RandomForestRegressor(random_state=42)
            model.fit(X, y)
            joblib.dump(model, f"{MODEL_DIR}/model_user.pkl")
            st.success("✅ Model trained on uploaded data!")

            rand_idx = np.random.randint(0, len(X))
            X_sample = X.iloc[[rand_idx]]
            st.write(f"### 🔍 Random Row (Index {rand_idx})")
            st.dataframe(X_sample)
            explain_row(model, X_sample)

            st.subheader("📈 Metrics on Uploaded Data")
            preds = model.predict(X)
            st.json(evaluate_model(model, X, y))
    else:
        st.info("👆 Upload a CSV to train your own model.")
