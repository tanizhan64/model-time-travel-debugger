# 🧠 Model-Time Travel Debugger

A professional, explainable machine learning app to compare and debug model versions using SHAP and feature drift tracking — no API keys needed.

---

## 🚀 Features

| Phase | Module | Description |
|-------|--------|-------------|
| ✅ 1 | **SHAP Viewer** | Visualize feature impact per row, per model version |
| ✅ 2 | **Prediction Difference Explainer** | Shows what changed and why the output is different |
| ✅ 3 | **Model Comparison + Drift Tracker** | Track MAE, RMSE, R² and feature distribution changes |
| ✅ 4 | **UI Controls + Retraining** | Fully interactive Streamlit app with model rebuild |

---

## 🗂️ Folder Structure

```
model-time-travel-debugger/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Install dependencies
├── .streamlit/config.toml  # Theme and UI config
├── data/
│   ├── housing_v1.csv
│   └── housing_v2.csv
└── models/
    ├── model_v1.pkl
    └── model_v2.pkl
```

---

## 📦 Install

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📈 Use Case

This tool helps:
- ✅ Explain tabular model predictions
- 🔄 Compare model versions (before/after retraining)
- ⚠️ Detect data drift
- 📊 Communicate feature importance without APIs

---

## 📤 Deploy

- Upload to [Streamlit Cloud](https://streamlit.io/cloud)
- Or push to GitHub, link it to deploy in minutes

---

## 👑 Built With

- `Streamlit`
- `scikit-learn`
- `SHAP`
- `pandas`, `numpy`, `matplotlib`

---

## 📘 License

MIT – Use freely and customize for your own modeling pipelines.
