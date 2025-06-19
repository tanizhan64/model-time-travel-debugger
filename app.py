# app.py — Separated Metrics + Drift Sections

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
        "RMSE": np.sqrt(mean_squared_error(y, preds)),  # Manual sqrt
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
    change = pred_v2 - pred_v1
    direction = "increased" if change > 0 else "decreased"
    pct_change = abs(change) / (abs(pred_v1) + 1e-8) * 100
    txt = f"🔎 The prediction has **{direction} by {pct_change:.2f}%**, from `{pred_v1:.2f}` to `{pred_v2:.2f}`.\n\n"
    txt += "Top changed feature importance between versions:\n"
    for feat, delta in top_features:
        arrow = "⬆️" if delta > 0 else "⬇️"
        txt += f"- {arrow} `{feat}` changed by `{delta:.3f}` SHAP points\n"
    return txt

st.title("🧠 Model-Time Travel Debugger")
st.sidebar.title("🔥 Highly Recommended")
st.sidebar.markdown("""
- Use built-in housing demo
- Select a model and row
- Explain button shows version differences
- Metrics and drift are in separate sections
- Retrain button refreshes models
""")

for ver in ["v1", "v2"]:
    if not os.path.exists(MODEL_PATHS[ver]):
        df = pd.read_csv(DATA_PATHS[ver])
        train_and_save_model(df, ver)

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
st.success(f"Prediction: `{pred:.2f}`")

explain_row(model, X_sample)

if st.button("🧠 Explain v1 vs v2 Shift"):
    df_v1 = pd.read_csv(DATA_PATHS["v1"])
    df_v2 = pd.read_csv(DATA_PATHS["v2"])
    X1, y1 = df_v1.drop(columns=["target"]), df_v1["target"]
    X2, y2 = df_v2.drop(columns=["target"]), df_v2["target"]
    model_v1 = joblib.load(MODEL_PATHS["v1"])
    model_v2 = joblib.load(MODEL_PATHS["v2"])

    pred_v1 = model_v1.predict(X_sample)[0]
    pred_v2 = model_v2.predict(X_sample)[0]

    explainer_v1 = shap.Explainer(model_v1)
    explainer_v2 = shap.Explainer(model_v2)
    shap_v1 = explainer_v1(X_sample)
    shap_v2 = explainer_v2(X_sample)

    diffs = shap_v2.values[0] - shap_v1.values[0]
    top_idx = np.argsort(np.abs(diffs))[::-1][:3]
    top_features = [(X.columns[i], diffs[i]) for i in top_idx]

    st.markdown("### 🗣️ Natural Explanation")
    st.info(get_explanation_text(pred_v1, pred_v2, top_features))

    metrics_v1 = evaluate_model(model_v1, X1, y1)
    metrics_v2 = evaluate_model(model_v2, X2, y2)
    drift_df = pd.DataFrame({
        "Feature": X1.columns,
        "Mean_v1": X1.mean().values,
        "Mean_v2": X2.mean().values,
        "ΔMean": X2.mean().values - X1.mean().values
    })

    with st.expander("📊 Metrics Comparison"):
        st.write("Model v1:", metrics_v1)
        st.write("Model v2:", metrics_v2)

    with st.expander("📉 Feature Drift"):
        st.dataframe(drift_df)

if st.button("🔁 Retrain v1 & v2 Models"):
    for ver in ["v1", "v2"]:
        df = pd.read_csv(DATA_PATHS[ver])
        train_and_save_model(df, ver)
    st.success("✅ Models retrained.")
