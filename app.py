from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import html

from business_logic import risk_bucket

app = Flask(__name__)

# ---------------------------
# Load models (repo root)
# ---------------------------
XGB_MODEL_PATH = "xgb_model.pkl"
DT_MODEL_PATH = "dt_model.pkl"

# Get expected input columns from model (pipeline)
EXPECTED_COLS = None
if hasattr(xgb_model, "feature_names_in_"):
    EXPECTED_COLS = list(xgb_model.feature_names_in_)

# Fallback if not found
if not EXPECTED_COLS:
    # Put your column list manually if needed
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

# ---------------------------
# Helpers
# ---------------------------
def guess_input_type(col: str):
    """
    Very simple rule-based guess:
    - *_flag, *_days, *_pct, *_kg, *_cbm, *_usd -> number
    - else -> text
    """
    c = col.lower()
    if c.endswith("_flag") or c.endswith("_days") or c.endswith("_pct") or c.endswith("_kg") or c.endswith("_cbm") or c.endswith("_usd"):
        return "number"
    return "text"

def coerce_value(col: str, val: str):
    """
    Convert numeric-looking fields to float/int.
    Keep text fields as string.
    Empty values:
      - numeric -> 0
      - text -> ""
    """
    t = guess_input_type(col)
    if val is None:
        val = ""
    val = val.strip()

    if t == "number":
        if val == "":
            return 0
        # allow integer/float
        try:
            if "." in val:
                return float(val)
            return int(val)
        except:
            return 0
    else:
        return val

def predict_from_df(df: pd.DataFrame):
    """
    Try XGBoost pipeline first, fallback to DecisionTree.
    """
    try:
        prob = float(xgb_model.predict_proba(df)[0][1])
        model_used = "XGBoost"
    except Exception:
        prob = float(dt_model.predict_proba(df)[0][1])
        model_used = "DecisionTree"

    bucket, action = risk_bucket(prob)
    return {
        "model_used": model_used,
        "sla_breach_probability": round(prob, 3),
        "risk_bucket": bucket,
        "business_action": action
    }

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "api": "Carrier SLA Risk Prediction API",
        "status": "running",
        "endpoints": {
            "ui": "/ui",
            "single_prediction_json": "/predict",
            "batch_prediction_json": "/predict/batch"
        }
    })

@app.route("/ui", methods=["GET"])
def ui():
    # Build form inputs dynamically
    inputs_html = []
    for col in EXPECTED_COLS:
        input_type = guess_input_type(col)
        # numeric inputs: step allows decimals
        step_attr = ' step="any"' if input_type == "number" else ""
        inputs_html.append(f"""
        <div class="row">
          <label for="{html.escape(col)}">{html.escape(col)}</label>
          <input id="{html.escape(col)}" name="{html.escape(col)}" type="{input_type}"{step_attr} placeholder="Enter {html.escape(col)}" />
        </div>
        """)

    page = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Carrier SLA Risk Prediction - UI</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      max-width: 980px;
      margin: 20px auto;
      padding: 0 16px;
    }}
    .card {{
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
      background: #fff;
    }}
    h2 {{ margin: 0 0 8px 0; }}
    .row {{
      display: grid;
      grid-template-columns: 300px 1fr;
      gap: 12px;
      align-items: center;
      padding: 6px 0;
    }}
    label {{ font-weight: 600; }}
    input {{
      padding: 8px;
      border: 1px solid #ccc;
      border-radius: 6px;
      width: 100%;
    }}
    .actions {{
      margin-top: 12px;
      display: flex;
      gap: 10px;
      align-items: center;
    }}
    button {{
      padding: 10px 14px;
      border: 1px solid #333;
      background: #111;
      color: #fff;
      border-radius: 6px;
      cursor: pointer;
    }}
    button.secondary {{
      background: #fff;
      color: #111;
    }}
    pre {{
      background: #f6f6f6;
      border: 1px solid #ddd;
      padding: 12px;
      border-radius: 8px;
      white-space: pre-wrap;
    }}
    .hint {{
      color: #444;
      font-size: 13px;
      margin-top: 6px;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h2>Carrier SLA Risk Prediction</h2>
    <div class="hint">
      Fill the fields and click <b>Predict</b>. Empty numeric fields will be treated as <b>0</b>.
    </div>
  </div>

  <div class="card">
    <form id="predForm">
      {''.join(inputs_html)}
      <div class="actions">
        <button type="button" onclick="submitPredict()">Predict</button>
        <button type="button" class="secondary" onclick="document.getElementById('predForm').reset(); document.getElementById('result').textContent='';">Clear</button>
      </div>
    </form>
  </div>

  <div class="card">
    <h3>Result</h3>
    <pre id="result"></pre>
  </div>

<script>
async function submitPredict() {{
  const resultEl = document.getElementById("result");
  resultEl.textContent = "Running...";

  const form = document.getElementById("predForm");
  const data = new FormData(form);

  // Convert form to JSON object
  const payload = {{}};
  for (const [k, v] of data.entries()) {{
    payload[k] = v;
  }}

  try {{
    const res = await fetch("/ui/predict", {{
      method: "POST",
      headers: {{"Content-Type":"application/json"}},
      body: JSON.stringify(payload)
    }});
    const text = await res.text();
    resultEl.textContent = text;
  }} catch (e) {{
    resultEl.textContent = "Error: " + e.message;
  }}
}}
</script>
</body>
</html>
"""
    return page

@app.route("/ui/predict", methods=["POST"])
def ui_predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid input"}), 400

    # Build one-row dict with correct types + correct column order
    row = {}
    for col in EXPECTED_COLS:
        row[col] = coerce_value(col, data.get(col, ""))

    df = pd.DataFrame([row])
    out = predict_from_df(df)
    return jsonify(out)

# JSON API endpoints remain available
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON. Send a JSON object."}), 400

    # Make sure all expected cols exist; missing -> 0 / ""
    row = {}
    for col in EXPECTED_COLS:
        row[col] = coerce_value(col, str(data.get(col, "")) if data.get(col, "") is not None else "")

    df = pd.DataFrame([row])
    out = predict_from_df(df)
    return jsonify(out)

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)
    if not isinstance(data, list) or len(data) == 0:
        return jsonify({"error": "Invalid JSON. Send a list of objects."}), 400

    df = pd.DataFrame(data)

    # Ensure expected columns exist
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = 0 if guess_input_type(col) == "number" else ""

    # Keep only expected order
    df = df[EXPECTED_COLS]

    # Try XGB then fallback
    try:
        probs = xgb_model.predict_proba(df)[:, 1]
        model_used = "XGBoost"
    except Exception:
        probs = dt_model.predict_proba(df)[:, 1]
        model_used = "DecisionTree"

    results = []
    for p in probs:
        p = float(p)
        bucket, action = risk_bucket(p)
        results.append({
            "model_used": model_used,
            "sla_breach_probability": round(p, 3),
            "risk_bucket": bucket,
            "business_action": action
        })

    return jsonify(results)

# Render / local entrypoint
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
