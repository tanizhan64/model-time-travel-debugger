# 🧠 Model-Time Travel Debugger – Phase 4: Save Reports + Classification

import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.utils.multiclass import type_of_target

# ------------------------------
# 📦 Setup
# ------------------------------
st.set_page_config(page_title="Model-Time Travel Debugger", layout="wide")

MODEL_DIR = "models"
EXAMPLE_DIR = "data"
UPLOAD_DIR = "user_data"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXAMPLE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

MODEL_PATHS = {f"v{i}": f"{MODEL_DIR}/model_v{i}.pkl" for i in [1, 2]}
DATA_PATHS = {f"v{i}": f"{UPLOAD_DIR}/upload_v{i}.csv" for i in [1, 2]}
TARGET_META = {f"v{i}": f"{UPLOAD_DIR}/target_v{i}.txt" for i in [1, 2]}
TYPE_META = {f"v{i}": f"{UPLOAD_DIR}/type_v{i}.txt" for i in [1, 2]}


# ------------------------------
# 🧠 Utilities
# ------------------------------
def detect_task_type(y):
    return "classification" if type_of_target(y) in ["binary", "multiclass"] else "regression"

def train_model(data, version, target_col):
    X = data.drop(columns=[target_col])
    y = data[target_col]
    task_type = detect_task_type(y)
    model = RandomForestClassifier(random_state=42) if task_type == "classification" else RandomForestRegressor(random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATHS[version])
    with open(TYPE_META[version], "w") as f:
        f.write(task_type)
    return model

def get_model_type(ver):
    return open(TYPE_META[ver]).read().strip() if os.path.exists(TYPE_META[ver]) else "regression"

def evaluate(model, X, y, task_type):
    preds = model.predict(X)
    if task_type == "classification":
        return {
            "Accuracy": accuracy_score(y, preds),
            "F1 Score": f1_score(y, preds, average="weighted")
        }
    return {
        "MAE": mean_absolute_error(y, preds),
        "RMSE": np.sqrt(mean_squared_error(y, preds)),
        "R2": r2_score(y, preds)
    }

def explain_row(model, X_sample):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    st.subheader("📊 SHAP Explanation")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

def get_data(ver):
    return pd.read_csv(DATA_PATHS[ver])

def get_target_col(ver, df):
    return open(TARGET_META[ver]).read().strip()


# ------------------------------
# 🧠 App Interface
# ------------------------------
st.title("🧠 Model-Time Travel Debugger (Phases 1–4 Complete)")
mode = st.radio("Choose Dataset Mode", ["📘 Example Dataset", "📁 Upload CSVs"])

# Load data
if mode == "📘 Example Dataset":
    for ver in ["v1", "v2"]:
        import shutil
        shutil.copyfile(f"{EXAMPLE_DIR}/housing_{ver}.csv", DATA_PATHS[ver])
        with open(TARGET_META[ver], "w") as f:
            f.write("target")
        train_model(pd.read_csv(DATA_PATHS[ver]), ver, "target")

if mode == "📁 Upload CSVs":
    for ver in ["v1", "v2"]:
        st.markdown(f"### 🔄 Upload v{ver.upper()}")
        uploaded = st.file_uploader(f"Upload CSV for {ver.upper()}", type=["csv"], key=ver)
        target_col = st.text_input(f"Target column for {ver.upper()}", key=f"target_{ver}")
        if uploaded and target_col:
            df = pd.read_csv(uploaded)
            if target_col in df.columns:
                df.to_csv(DATA_PATHS[ver], index=False)
                with open(TARGET_META[ver], "w") as f:
                    f.write(target_col)
                train_model(df, ver, target_col)
                st.success(f"✅ {ver.upper()} uploaded and model trained")
            else:
                st.error(f"'{target_col}' not found in CSV columns")

# Validate data
if not all([os.path.exists(DATA_PATHS[v]) and os.path.exists(TARGET_META[v]) and os.path.exists(TYPE_META[v]) for v in ["v1", "v2"]]):
    st.warning("Please upload both CSVs and enter valid targets to proceed.")
    st.stop()

# ------------------------------
# 🎯 Predict + SHAP
# ------------------------------
selected_version = st.selectbox("Select Model Version", ["v1", "v2"])
df = get_data(selected_version)
target_col = get_target_col(selected_version, df)
task_type = get_model_type(selected_version)
model = joblib.load(MODEL_PATHS[selected_version])
X = df.drop(columns=[target_col])
y = df[target_col]

row = st.slider("Pick Row Index", 0, len(X)-1, 0)
X_sample = X.iloc[[row]]

st.write("### 🔍 Input Features")
st.dataframe(X_sample)

pred = model.predict(X_sample)[0]
st.write("### 📈 Prediction")
st.success(f"Model predicts: `{pred}`")

explain_row(model, X_sample)


# ------------------------------
# 📊 Compare v1 vs v2
# ------------------------------
if st.button("🧠 Explain v1 vs v2"):
    df1, df2 = get_data("v1"), get_data("v2")
    t1, t2 = get_target_col("v1", df1), get_target_col("v2", df2)
    model1, model2 = joblib.load(MODEL_PATHS["v1"]), joblib.load(MODEL_PATHS["v2"])
    X_sample = X.iloc[[row]]
    shap1, shap2 = shap.Explainer(model1)(X_sample), shap.Explainer(model2)(X_sample)
    diff = shap2.values[0] - shap1.values[0]
    top = np.argsort(np.abs(diff))[::-1][:3]
    for i in top:
        f = X.columns[i]
        st.write(f"**{f}**: `{shap1.values[0][i]:.3f}` → `{shap2.values[0][i]:.3f}`")


# ------------------------------
# 📏 Metrics + Drift
# ------------------------------
if st.button("📈 View Metrics + Drift"):
    df1 = get_data("v1")
    df2 = get_data("v2")
    t1 = get_target_col("v1", df1)
    t2 = get_target_col("v2", df2)
    task1 = get_model_type("v1")
    task2 = get_model_type("v2")
    m1 = joblib.load(MODEL_PATHS["v1"])
    m2 = joblib.load(MODEL_PATHS["v2"])
    x1, y1 = df1.drop(columns=[t1]), df1[t1]
    x2, y2 = df2.drop(columns=[t2]), df2[t2]

    st.markdown("### 📏 Model Metrics")
    st.markdown("#### v1")
    for k, v in evaluate(m1, x1, y1, task1).items():
        st.markdown(f"- **{k}**: `{v:.4f}`")
    st.markdown("#### v2")
    for k, v in evaluate(m2, x2, y2, task2).items():
        st.markdown(f"- **{k}**: `{v:.4f}`")

    st.markdown("### 🔄 Feature Drift")
    drift = pd.DataFrame({
        "Feature": x1.columns,
        "Mean_v1": x1.mean(),
        "Mean_v2": x2.mean(),
        "ΔMean": x2.mean() - x1.mean()
    })
    st.dataframe(drift)

# ------------------------------
# 💾 Save SHAP + Drift Reports
# ------------------------------
if st.button("💾 Save SHAP + Drift Reports"):
    for ver in ["v1", "v2"]:
        model = joblib.load(MODEL_PATHS[ver])
        df = get_data(ver)
        target = get_target_col(ver, df)
        X = df.drop(columns=[target])
        X_sample = X.iloc[[row]]
        explainer = shap.Explainer(model)
        shap_vals = explainer(X_sample)
        shap_df = pd.DataFrame({
            "Feature": X.columns,
            "SHAP": shap_vals.values[0]
        })
        shap_df.to_csv(f"{REPORT_DIR}/shap_{ver}.csv", index=False)

    # ✅ FIX: Redefine x1 and x2 locally here
    df1 = get_data("v1")
    df2 = get_data("v2")
    t1 = get_target_col("v1", df1)
    t2 = get_target_col("v2", df2)
    x1, x2 = df1.drop(columns=[t1]), df2.drop(columns=[t2])

    drift = pd.DataFrame({
        "Feature": x1.columns,
        "Mean_v1": x1.mean(),
        "Mean_v2": x2.mean(),
        "ΔMean": x2.mean() - x1.mean()
    })
    drift.to_csv(f"{REPORT_DIR}/feature_drift.csv", index=False)
    st.success("✅ SHAP + Drift reports saved to /reports/")

# ------------------------------
# 🔁 Retrain Models
# ------------------------------
if st.button("🔁 Retrain Models"):
    for ver in ["v1", "v2"]:
        df = get_data(ver)
        target = get_target_col(ver, df)
        train_model(df, ver, target)
    st.success("✅ Models retrained from current data.")
