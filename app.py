
# 🧠 Model-Time Travel Debugger (Upload Inputs on Main Page)

import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
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

def get_data(ver):
    return pd.read_csv(UPLOAD_PATHS[ver]) if os.path.exists(UPLOAD_PATHS[ver]) else pd.read_csv(EXAMPLE_PATHS[ver])

def get_target(ver, df):
    return open(TARGET_META[ver]).read().strip() if os.path.exists(TARGET_META[ver]) else "target"

st.title("🧠 Model-Time Travel Debugger (Main Page Upload)")

mode = st.radio("Choose Mode", ["📘 Example Dataset", "📁 Upload CSVs"])

uploaded = {}
target_names = {}

if mode == "📁 Upload CSVs":
    st.markdown("### 📤 Upload CSVs and Specify Target Columns")

    for ver in ["v1", "v2"]:
        st.markdown(f"#### 🔹 Version {ver.upper()}")
        uploaded[ver] = st.file_uploader(f"Upload CSV for {ver.upper()}", type=["csv"], key=f"u_{ver}")
        target_names[ver] = st.text_input(f"Target column for {ver.upper()}", key=f"t_{ver}")

        if uploaded[ver] and target_names[ver]:
            df = pd.read_csv(uploaded[ver])
            if target_names[ver] in df.columns:
                df.to_csv(UPLOAD_PATHS[ver], index=False)
                with open(TARGET_META[ver], "w") as f:
                    f.write(target_names[ver])
                train_and_save_model(df, ver, target_names[ver])
                st.success(f"{ver.upper()} uploaded and trained on `{target_names[ver]}`.")
            else:
                st.error(f"❌ '{target_names[ver]}' not found in columns: {list(df.columns)}")

# Block app if uploads not ready
if mode == "📁 Upload CSVs" and not all(os.path.exists(UPLOAD_PATHS[v]) and os.path.exists(TARGET_META[v]) for v in ["v1", "v2"]):
    st.warning("📛 Please upload both CSVs and enter valid target column names.")
    st.stop()

selected_version = st.selectbox("Select Model Version", ["v1", "v2"])
df = get_data(selected_version)
target = get_target(selected_version, df)
model = joblib.load(MODEL_PATHS[selected_version])
X = df.drop(columns=[target])
y = df[target]

row_idx = st.slider("Pick Row Index", 0, len(X) - 1, 0)
X_sample = X.iloc[[row_idx]]

st.write("### 🔍 Input Features")
st.dataframe(X_sample)

st.write("### 📈 Prediction")
pred = model.predict(X_sample)[0]
st.success(f"Model {selected_version} predicts: `{pred:.2f}`")

explain_row(model, X_sample)

if st.button("🧠 Explain v1 vs v2 Shift"):
    model_v1 = joblib.load(MODEL_PATHS["v1"])
    model_v2 = joblib.load(MODEL_PATHS["v2"])
    df_v1 = get_data("v1")
    df_v2 = get_data("v2")
    t1 = get_target("v1", df_v1)
    t2 = get_target("v2", df_v2)
    pred1 = model_v1.predict(X_sample)[0]
    pred2 = model_v2.predict(X_sample)[0]
    shap1 = shap.Explainer(model_v1)(X_sample)
    shap2 = shap.Explainer(model_v2)(X_sample)
    delta = shap2.values[0] - shap1.values[0]
    top = np.argsort(np.abs(delta))[::-1][:3]
    for i in top:
        st.write(f"**{X.columns[i]}**: `{shap1.values[0][i]:.3f}` → `{shap2.values[0][i]:.3f}`")

if st.button("📈 Metrics + Drift"):
    df1 = get_data("v1")
    df2 = get_data("v2")
    t1 = get_target("v1", df1)
    t2 = get_target("v2", df2)
    m1 = joblib.load(MODEL_PATHS["v1"])
    m2 = joblib.load(MODEL_PATHS["v2"])
    x1, y1 = df1.drop(columns=[t1]), df1[t1]
    x2, y2 = df2.drop(columns=[t2]), df2[t2]
    st.markdown("#### Model v1")
    for k, v in evaluate_model(m1, x1, y1).items():
        st.markdown(f"- **{k}**: `{v:.4f}`")
    st.markdown("#### Model v2")
    for k, v in evaluate_model(m2, x2, y2).items():
        st.markdown(f"- **{k}**: `{v:.4f}`")
    drift_df = pd.DataFrame({
        "Feature": x1.columns,
        "Mean_v1": x1.mean(),
        "Mean_v2": x2.mean(),
        "ΔMean": x2.mean() - x1.mean()
    })
    st.dataframe(drift_df)
