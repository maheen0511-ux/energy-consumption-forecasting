# ⚡ Energy Consumption Forecaster
### End-to-End ML Project | Seasonal Decomposition + XGBoost + Flask Web UI

---

## 📁 Project Structure

```
energy-forecaster/
├── energy_forecaster.ipynb   ← Full ML pipeline (run this first!)
├── app.py                    ← Flask web server
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
├── templates/
│   └── index.html            ← Web UI (HTML/CSS/JS)
└── [generated after notebook run]
    ├── energy_model.pkl      ← Trained XGBoost model
    ├── model_metadata.json   ← Model performance stats
    ├── eda_overview.png      ← EDA charts
    ├── eda_heatmap.png       ← Heatmap
    ├── eda_correlation.png   ← Correlation plot
    ├── stl_decomposition.png ← STL decomposition
    └── model_evaluation.png  ← Model comparison charts
```

---

## 🚀 Quick Start

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the Jupyter Notebook
Open and run all cells to:
- Generate synthetic energy dataset
- Perform EDA & seasonal decomposition
- Engineer 35+ features
- Train & compare models (Ridge, Random Forest, XGBoost)
- Save `energy_model.pkl`

```bash
jupyter notebook energy_forecaster.ipynb
```

### Step 3 — Start the Web App
```bash
python app.py
```

### Step 4 — Open in Browser
```
http://localhost:5000
```

---

## 🧠 ML Pipeline

| Stage | Details |
|-------|---------|
| **Dataset** | 2-year synthetic hourly energy data (17,520 rows) |
| **Decomposition** | STL (Seasonal-Trend-Loess) with period=365 |
| **Features** | Time cyclical, lag (1h–168h), rolling stats, temperature derived |
| **Models** | Baseline → Ridge → Random Forest → XGBoost |
| **Best Model** | XGBoost (n_estimators=500, early stopping) |
| **Evaluation** | MAE, RMSE, R², MAPE + residual ACF/PACF check |

---

## 🌐 Web UI Features

- **Real-time prediction** via Flask REST API
- **Drag sliders** for hour and temperature
- **Energy level** indicator (Low / Moderate / High)
- **Daily & monthly** consumption estimates
- **Heating/Cooling degree days** display
- **Model metrics** in the header stats bar
- **Dark industrial** aesthetic with animated grid background

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/predict` | POST | Get energy prediction |
| `/health` | GET | Model status & metadata |

**POST /predict body:**
```json
{
  "hour": 18,
  "day_of_week": 1,
  "day_of_month": 15,
  "month": 7,
  "temperature": 28.5
}
```

**Response:**
```json
{
  "prediction": 74.32,
  "level": "HIGH",
  "level_color": "#F44336",
  "insight": "15.2% above the historical average",
  "daily_estimate": 1783.7,
  "monthly_estimate": 53511.0
}
```

---

## 🔧 Customization Tips

- **Real data**: Replace the synthetic data generator with a CSV loader
  (e.g., UCI Household Power, PJM Energy dataset)
- **More models**: Add LightGBM, LSTM for comparison
- **Holidays**: Integrate `holidays` library for `is_holiday` feature
- **Deployment**: Host on Render / Railway / Heroku for free

---

*Built with Python · XGBoost · statsmodels · Flask · Vanilla JS*
