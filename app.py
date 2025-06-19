# Extended version with Upload + Retrain

import pandas as pd
import numpy as np
import shap
import joblib
import streamlit as st
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_DIR = "models"
USER_DATA_PATH = "data/user_upload.csv"
USER_MODEL_PATH = f"{MODEL_DIR}/model_user.pkl"

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

st.title("🧠 Model-Time Travel Debugger (With Upload Support)")

# -----------------------------
# Upload section
# -----------------------------
st.header("📤 Upload Custom Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df_upload = pd.read_csv(uploaded_file)
    df_upload.to_csv(USER_DATA_PATH, index=False)
    st.success("✅ File uploaded and saved.")

    st.write("### Preview Uploaded Data")
    st.dataframe(df_upload.head())

    # Target column selection
    with st.form("target_form"):
        target_col = st.selectbox("🎯 Select Target Column", df_upload.columns)
        submitted = st.form_submit_button("Train Model on Uploaded Data")

    if submitted:
        X = df_upload.drop(columns=[target_col])
        y = df_upload[target_col]

        # Train model
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        joblib.dump(model, USER_MODEL_PATH)
        st.success("✅ Model trained and saved as model_user.pkl")

        # Show SHAP explanation for random row
        st.subheader("🔍 SHAP Explanation (Random Row)")
        sample_idx = np.random.randint(0, len(X))
        row = X.iloc[[sample_idx]]
        st.write("Row Index:", sample_idx)
        st.write(row)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(row)
        shap.initjs()
        st_shap = shap.force_plot(explainer.expected_value, shap_values, row, matplotlib=False)
        st.components.v1.html(st_shap.html(), height=300)

        # Metrics on full uploaded dataset
        preds = model.predict(X)
        st.subheader("📈 Model Evaluation Metrics")
        st.write({
            "MAE": mean_absolute_error(y, preds),
            "RMSE": mean_squared_error(y, preds) ** 0.5,
            "R2": r2_score(y, preds)
        })
else:
    st.info("👆 Upload a CSV file to get started.")
