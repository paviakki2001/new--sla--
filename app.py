from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import html

from business_logic import risk_bucket

app = Flask(__name__)

# =====================================================
# LOAD MODELS (FROM ROOT FOLDER - IMPORTANT)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

XGB_MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.pkl")
DT_MODEL_PATH  = os.path.join(BASE_DIR, "dt_model.pkl")

try:
    xgb_model = joblib.load(XGB_MODEL_PATH)
    print("XGB model loaded successfully")
except Exception as e:
    print("Error loading XGB model:", e)
    xgb_model = None

try:
    dt_model = joblib.load(DT_MODEL_PATH)
    print("DT model loaded successfully")
except Exception as e:
    print("Error loading DT model:", e)
    dt_model = None


# =====================================================
# EXPECTED COLUMNS
# =====================================================

EXPECTED_COLS = None

if xgb_model is not None and hasattr(xgb_model, "feature_names_in_"):
    EXPECTED_COLS = list(xgb_model.feature_names_in_)

if not EXPECTED_COLS:
    EXPECTED_COLS = [
        "shipment_id", "carrier", "shipping_mode", "region",
        "origin_country", "destination_country",
        "shipment_weight_kg", "shipment_volume_cbm",
        "priority_flag", "fragile_flag", "temperature_control_flag",
        "planned_delivery_days", "actual_delivery_days", "delivery_delay_days",
        "shipping_cost_usd", "fuel_surcharge_pct",
        "customs_delay_flag", "weather_disruption_flag",
        "shipment_value_usd", "insurance_flag"
    ]


# =====================================================
# HELPERS
# =====================================================

def guess_input_type(col):
    c = col.lower()
    if (
        c.endswith("_flag")
        or c.endswith("_days")
        or c.endswith("_pct")
        or c.endswith("_kg")
        or c.endswith("_cbm")
        or c.endswith("_usd")
    ):
        return "number"
    return "text"


def coerce_value(col, val):
    if val is None:
        val = ""

    val = str(val).strip()
    t = guess_input_type(col)

    if t == "number":
        if val == "":
            return 0
        try:
            return float(val)
        except:
            return 0
    else:
        return val


def predict_from_df(df):

    if xgb_model is not None:
        try:
            prob = float(xgb_model.predict_proba(df)[0][1])
            model_used = "XGBoost"
        except:
            prob = float(dt_model.predict_proba(df)[0][1])
            model_used = "DecisionTree"
    else:
        prob = float(dt_model.predict_proba(df)[0][1])
        model_used = "DecisionTree"

    bucket, action = risk_bucket(prob)

    return {
        "model_used": model_used,
        "sla_breach_probability": round(prob, 3),
        "risk_bucket": bucket,
        "business_action": action
    }


# =====================================================
# ROUTES
# =====================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "api": "Carrier SLA Risk Prediction API",
        "status": "running",
        "ui": "/ui",
        "single_prediction": "/predict",
        "batch_prediction": "/predict/batch"
    })


@app.route("/ui", methods=["GET"])
def ui():

    inputs_html = []

    for col in EXPECTED_COLS:
        input_type = guess_input_type(col)
        step_attr = ' step="any"' if input_type == "number" else ""

        inputs_html.append(f"""
        <div class="row">
            <label>{html.escape(col)}</label>
            <input name="{html.escape(col)}" type="{input_type}"{step_attr} />
        </div>
        """)

    page = f"""
    <html>
    <head>
    <title>Carrier SLA Risk Prediction</title>
    <style>
    body {{
        font-family: Arial;
        max-width: 900px;
        margin: auto;
        padding: 20px;
    }}
    .row {{
        display: grid;
        grid-template-columns: 300px 1fr;
        gap: 10px;
        margin-bottom: 8px;
    }}
    input {{
        padding: 6px;
    }}
    button {{
        padding: 8px 12px;
        margin-top: 10px;
    }}
    pre {{
        background: #f4f4f4;
        padding: 10px;
        margin-top: 20px;
    }}
    </style>
    </head>
    <body>

    <h2>Carrier SLA Risk Prediction</h2>

    <form id="form">
        {''.join(inputs_html)}
        <button type="button" onclick="predict()">Predict</button>
    </form>

    <pre id="result"></pre>

    <script>
    async function predict() {{
        const form = document.getElementById("form");
        const data = new FormData(form);
        let obj = {{}};
        for (let [k, v] of data.entries()) {{
            obj[k] = v;
        }}

        const res = await fetch("/predict", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify(obj)
        }});

        const text = await res.text();
        document.getElementById("result").textContent = text;
    }}
    </script>

    </body>
    </html>
    """

    return page


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON"}), 400

    row = {}
    for col in EXPECTED_COLS:
        row[col] = coerce_value(col, data.get(col))

    df = pd.DataFrame([row])

    result = predict_from_df(df)

    return jsonify(result)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():

    data = request.get_json()

    if not isinstance(data, list):
        return jsonify({"error": "Send list of JSON objects"}), 400

    df = pd.DataFrame(data)

    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = 0 if guess_input_type(col) == "number" else ""

    df = df[EXPECTED_COLS]

    if xgb_model is not None:
        try:
            probs = xgb_model.predict_proba(df)[:, 1]
            model_used = "XGBoost"
        except:
            probs = dt_model.predict_proba(df)[:, 1]
            model_used = "DecisionTree"
    else:
        probs = dt_model.predict_proba(df)[:, 1]
        model_used = "DecisionTree"

    results = []

    for p in probs:
        bucket, action = risk_bucket(float(p))
        results.append({
            "model_used": model_used,
            "sla_breach_probability": round(float(p), 3),
            "risk_bucket": bucket,
            "business_action": action
        })

    return jsonify(results)


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
