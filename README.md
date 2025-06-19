# 🧠 Model-Time Travel Debugger (Streamlit + SHAP)

A production-grade machine learning debugger built with **Streamlit**, **scikit-learn**, and **SHAP** to analyze and compare different versions of ML models on tabular data.

---

## 📌 Overview

This app allows you to:
- 🔍 **Inspect model predictions** using SHAP explainability
- 🔄 **Compare two model versions (v1 and v2)**
- 📈 **Visualize prediction shifts and key driver changes**
- 📊 **Track model performance metrics (MAE, RMSE, R²)**
- ⚠️ **Detect feature drift (data quality change over time)**
- 🔁 **Retrain models inside the UI**
- 🧠 Run without external APIs (fully local, free to use)

---

## 📂 Folder Structure

```
model-time-travel-debugger/
├── app.py                  # 🚀 Main Streamlit app
├── requirements.txt        # 📦 Package dependencies
├── README.md               # 📘 You're here!
├── .streamlit/
│   └── config.toml         # 🎨 UI theme
├── data/
│   ├── housing_v1.csv      # 📊 Training data for model v1
│   └── housing_v2.csv      # 📊 Training data for model v2
└── models/
    ├── model_v1.pkl        # 🧠 Trained model v1 (auto created)
    └── model_v2.pkl        # 🧠 Trained model v2 (auto created)
```

---

## 🚀 How to Run

### 🔧 Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

> On first run, models will be trained and saved to `/models/`

---

## 🌍 Deploy to Streamlit Cloud

1. Go to: https://streamlit.io/cloud
2. Click **“New App”**
3. Set repo to: `tanizhan64/model-time-travel-debugger`
4. Set main file: `app.py`
5. Click **Deploy**

---

## 🔬 Phase Breakdown

| Phase | Module | Description |
|-------|--------|-------------|
| ✅ 1 | **Prediction + SHAP Viewer** | Interactive row viewer with feature importance |
| ✅ 2 | **Prediction Comparison Agent** | Compares predictions across model versions |
| ✅ 3 | **Metrics + Drift Tracker** | Shows MAE, RMSE, R² and feature drift stats |
| ✅ 4 | **UI + Retrain Panel** | Lets user retrain models and view results live |

---

## 📈 Example Metrics Output

```text
Model v1:
  MAE  = 0.38
  RMSE = 0.59
  R²   = 0.83

Model v2:
  MAE  = 0.35
  RMSE = 0.54
  R²   = 0.86
```

---

## ⚠️ Feature Drift Output

| Feature   | Mean_v1 | Mean_v2 | Mean_Diff | Std_Diff |
|-----------|---------|---------|-----------|----------|
| AveRooms  | 5.38    | 6.01    | 0.63      | 0.21     |
| MedInc    | 3.21    | 3.28    | 0.07      | 0.04     |

> This helps detect if your model is using shifted data or if retraining had effects due to new distributions.

---

## 🎯 Use Cases

- ✅ Model upgrade QA
- ✅ Local LLM tabular assistants
- ✅ Drift-aware retraining pipelines
- ✅ Explainable AI (XAI) demos
- ✅ Resume / portfolio project

---

## 🛠 Built With

- [Streamlit](https://streamlit.io)
- [scikit-learn](https://scikit-learn.org)
- [SHAP](https://github.com/slundberg/shap)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [joblib](https://joblib.readthedocs.io/)

---

## 📘 License

MIT License — use freely, credit appreciated. Fork for your own models, add upload support or plug into your MLOps stack!

---

## ✨ Author

Created by [@tanizhan64](https://github.com/tanizhan64) — your friendly AI time traveler 🧠🕰️
