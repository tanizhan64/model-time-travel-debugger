# 🧠 Model-Time Travel Debugger

A professional-grade Streamlit app to **debug, compare, and explain prediction shifts** between two versions of a machine learning model. Visualize SHAP explanations, inspect metrics, monitor data drift, and retrain models — all from one clean UI.

---

## 🚀 Features

| Section                     | What It Does                                                                 |
|-----------------------------|------------------------------------------------------------------------------|
| 🔍 **Row Picker**            | Select a row from your dataset to analyze                                   |
| 📈 **Prediction**            | View model output for the selected row                                      |
| 📊 **SHAP Explanation**      | Interactive waterfall plot to explain model reasoning                       |
| 🧠 **Explain v1 vs v2 Shift** | Compare predictions and SHAP explanations across model versions              |
| 📏 **Metrics + Drift (button)** | Check performance (MAE, RMSE, R²) and inspect feature distribution shifts    |
| 🔁 **Retrain Models**         | Rebuild v1 and v2 from updated CSVs                                         |

---

## 📂 Project Structure

```
model-time-travel-debugger/
├── app.py                          # Main Streamlit app (use final version)
├── models/
│   ├── model_v1.pkl
│   └── model_v2.pkl
├── data/
│   ├── housing_v1.csv              # Version 1 of dataset
│   └── housing_v2.csv              # Version 2 of dataset
├── requirements.txt
└── .streamlit/
    └── config.toml
```

---

## 🧪 Sample Data (Structure)

Both `housing_v1.csv` and `housing_v2.csv` must contain:

```csv
feature_1, feature_2, ..., feature_n, target
```

- `target` is the value the model learns to predict
- Feature names can vary but must match between versions

---

## 🛠 Setup Instructions

### 🔹 Local

```bash
git clone https://github.com/yourusername/model-time-travel-debugger.git
cd model-time-travel-debugger
pip install -r requirements.txt
streamlit run app.py
```

### 🔹 Streamlit Cloud

1. Push to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click "New App"
4. Select your repo and `app.py`
5. 🎉 You're live!

---

## 📊 Tech Stack

- **Frontend**: Streamlit
- **Modeling**: scikit-learn (RandomForestRegressor)
- **Explainability**: SHAP (`shap.Explainer`)
- **Metrics**: MAE, RMSE, R²

---

## 🧠 Example Use Cases

- Debug model changes in production
- Visualize data drift and feature impact
- Explain ML results to stakeholders
- Demo ML workflows in interviews
- Build trust in model decisions

---

## ✅ Coming Soon (Ideas to Expand)

- [ ] Upload your own CSVs directly
- [ ] Add support for classification tasks
- [ ] Save explanations and drift reports
- [ ] Add model version history and leaderboard

---

## 👤 Author

Built by [@tanizhan64](https://github.com/tanizhan64) with ✨ guidance from Infinity King (AI teammate)

---

## 📜 License

MIT License — use, fork, ship, and scale freely.
