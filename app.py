"""
⚡ Energy Consumption Forecaster — Flask Web Application
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__)

# ── Load model at startup ──────────────────────────────────────────
MODEL_PATH    = 'energy_model.pkl'
METADATA_PATH = 'model_metadata.json'

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Please run the Jupyter notebook first to train and save the model."
        )
    with open(MODEL_PATH, 'rb') as f:
        pkg = pickle.load(f)
    return pkg

def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            return json.load(f)
    return {}

try:
    model_package = load_model()
    model    = model_package['model']
    features = model_package['features']
    metadata = model_package.get('metadata', load_metadata())
    print(f"✅ Model loaded — {len(features)} features | R²={metadata.get('r2','N/A')}")
except FileNotFoundError as e:
    print(f"⚠️  {e}")
    model, features, metadata = None, [], {}


# ── Feature builder (mirrors notebook logic) ──────────────────────

def build_features(hour, day_of_week, day_of_month, month, temperature,
                   is_weekend, recent_energy=None):
    """
    Build the full feature vector from user inputs.
    For lag/rolling features we use proxies based on a plausible
    energy history generated from time-of-day and seasonal patterns.
    """
    # Generate plausible 'recent history' if not provided
    if recent_energy is None:
        base = 50 + 20 * np.sin(2 * np.pi * month / 12 - np.pi / 2)
        daily_curve = 10 * np.sin(2 * np.pi * (np.arange(24)) / 24 - np.pi)
        recent_history = np.array([
            base + daily_curve[h % 24] + 0.3 * abs(temperature - 18)
            for h in range(hour - 168, hour)
        ])
        recent_history = np.clip(recent_history, 10, None)
    else:
        recent_history = np.array(recent_energy)

    def lag(h):
        idx = -h if h > 0 else -1
        return float(recent_history[idx]) if abs(idx) <= len(recent_history) else float(recent_history[-1])

    def roll_mean(h):
        return float(recent_history[-h:].mean()) if h <= len(recent_history) else float(recent_history.mean())

    def roll_std(h):
        return float(recent_history[-h:].std()) if h <= len(recent_history) else float(recent_history.std())

    quarter = (month - 1) // 3 + 1
    week_of_year = min(52, (day_of_month + (month - 1) * 30) // 7 + 1)

    feat = {
        'hour': hour,
        'day_of_week': day_of_week,
        'day_of_month': day_of_month,
        'month': month,
        'quarter': quarter,
        'week_of_year': week_of_year,
        'is_weekend': int(is_weekend),
        'is_holiday': 0,

        # Cyclical
        'hour_sin':  np.sin(2 * np.pi * hour / 24),
        'hour_cos':  np.cos(2 * np.pi * hour / 24),
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),
        'dow_sin':   np.sin(2 * np.pi * day_of_week / 7),
        'dow_cos':   np.cos(2 * np.pi * day_of_week / 7),

        # Lags
        'lag_1h':   lag(1),
        'lag_2h':   lag(2),
        'lag_3h':   lag(3),
        'lag_6h':   lag(6),
        'lag_12h':  lag(12),
        'lag_24h':  lag(24),
        'lag_48h':  lag(48),
        'lag_168h': lag(168),

        # Rolling stats
        'roll_mean_6h':   roll_mean(6),
        'roll_std_6h':    roll_std(6),
        'roll_mean_12h':  roll_mean(12),
        'roll_std_12h':   roll_std(12),
        'roll_mean_24h':  roll_mean(24),
        'roll_std_24h':   roll_std(24),
        'roll_mean_48h':  roll_mean(48),
        'roll_std_48h':   roll_std(48),
        'roll_mean_168h': roll_mean(168),
        'roll_std_168h':  roll_std(168),

        # Temperature derived
        'temp_lag_1h':    temperature,
        'temp_roll_6h':   temperature,
        'temp_squared':   temperature ** 2,
        'heating_degree': max(18 - temperature, 0),
        'cooling_degree': max(temperature - 24, 0),
    }
    return feat


# ── Routes ──────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', metadata=metadata)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Run the notebook first.'}), 503

    try:
        data = request.get_json()

        hour         = int(data['hour'])
        day_of_week  = int(data['day_of_week'])
        day_of_month = int(data['day_of_month'])
        month        = int(data['month'])
        temperature  = float(data['temperature'])
        is_weekend   = int(day_of_week >= 5)

        feat_dict = build_features(
            hour=hour,
            day_of_week=day_of_week,
            day_of_month=day_of_month,
            month=month,
            temperature=temperature,
            is_weekend=is_weekend,
        )

        input_df   = pd.DataFrame([feat_dict])[features]
        prediction = float(model.predict(input_df)[0])
        prediction = max(0, round(prediction, 2))

        # Contextual insight
        avg = metadata.get('target_mean', 65)
        diff_pct = ((prediction - avg) / avg) * 100
        trend    = "above" if diff_pct > 0 else "below"
        insight  = f"{abs(diff_pct):.1f}% {trend} the historical average"

        # Level label
        if prediction < avg * 0.75:
            level, color = "LOW", "#4CAF50"
        elif prediction < avg * 1.1:
            level, color = "MODERATE", "#FF9800"
        else:
            level, color = "HIGH", "#F44336"

        return jsonify({
            'prediction': prediction,
            'level': level,
            'level_color': color,
            'insight': insight,
            'daily_estimate': round(prediction * 24, 1),
            'monthly_estimate': round(prediction * 24 * 30, 0),
        })

    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'features': len(features),
        'metadata': metadata
    })


if __name__ == '__main__':
    print("\n⚡ Energy Forecaster Web App")
    print("   URL: http://localhost:5000")
    print("   Press CTRL+C to quit\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
