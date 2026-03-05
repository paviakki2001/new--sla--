
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

from business_logic import risk_bucket

app = Flask(__name__)

# ---------------------------
# Model paths (choose ONE set)
# ---------------------------
# If your .pkl files are in the SAME folder as app.py:
XGB_MODEL_PATH = "xgb_model.pkl"
DT_MODEL_PATH = "dt_model.pkl"

# If your .pkl files are inside models/ folder, use this instead:
# XGB_MODEL_PATH = "models/xgb_model.pkl"
# DT_MODEL_PATH = "models/dt_model.pkl"

# Load models
xgb_model = joblib.load(XGB_MODEL_PATH)
dt_model = joblib.load(DT_MODEL_PATH)

# ---------------------------
# Expected features (must match your trained pipeline)
# ---------------------------
EXPECTED_COLUMNS = [
    "shipment_id", "carrier", "shipping_mode", "region",
    "origin_country", "destination_country",
    "shipment_weight_kg", "shipment_volume_cbm",
    "priority_flag", "fragile_flag", "temperature_control_flag",
    "planned_delivery_days", "actual_delivery_days", "delivery_delay_days",
    "shipping_cost_usd", "fuel_surcharge_pct",
    "customs_delay_flag", "weather_disruption_flag",
    "shipment_value_usd", "insurance_flag"
]

NUMERIC_FIELDS = {
    "shipment_weight_kg", "shipment_volume_cbm",
    "priority_flag", "fragile_flag", "temperature_control_flag",
    "planned_delivery_days", "actual_delivery_days", "delivery_delay_days",
    "shipping_cost_usd", "fuel_surcharge_pct",
    "customs_delay_flag", "weather_disruption_flag",
    "shipment_value_usd", "insurance_flag"
}

TEXT_FIELDS = [c for c in EXPECTED_COLUMNS if c not in NUMERIC_FIELDS]


# ---------------------------
# Home route
# ---------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "api": "Carrier SLA Risk Prediction API",
        "status": "running",
        "endpoints": {
            "ui": "/ui",
            "single_prediction": "/predict",
            "batch_prediction": "/predict/batch"
        }
    })


# ---------------------------
# Simple + Professional HTML Form UI
# ---------------------------
@app.route("/ui", methods=["GET"])
def ui():
    rows = ""

    # Text fields first
    for f in TEXT_FIELDS:
        rows += f"""
        <tr>
          <td class="k">{f}</td>
          <td><input class="in" type="text" id="{f}" placeholder="Enter {f}" /></td>
        </tr>
        """

    # Numeric fields
    for f in sorted(NUMERIC_FIELDS):
        rows += f"""
        <tr>
          <td class="k">{f}</td>
          <td><input class="in" type="number" step="any" id="{f}" placeholder="Enter {f}" /></td>
        </tr>
        """

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Carrier SLA Risk Prediction - UI</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 980px; margin: 20px; }}
    .box {{ border: 1px solid #ddd; padding: 14px; border-radius: 6px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td {{ padding: 8px; border-bottom: 1px solid #eee; vertical-align: top; }}
    .k {{ width: 280px; font-weight: 600; }}
    .in {{ width: 100%; padding: 8px; box-sizing: border-box; }}
    button {{ padding: 10px 14px; margin-right: 8px; cursor: pointer; }}
    pre {{ background: #f6f6f6; padding: 10px; border: 1px solid #ddd; white-space: pre-wrap; }}
    .muted {{ color: #666; font-size: 13px; }}
  </style>
</head>
<body>
  <h2>Carrier SLA Risk Prediction</h2>
  <p class="muted">
    Fill the inputs and click <b>Predict</b>. This UI accepts any values you enter and sends a JSON request to the API.
  </p>

  <div class="box">
    <table>
      {rows}
    </table>

    <br/>
    <button onclick="predictSingle()">Predict</button>
    <button onclick="clearAll()">Clear</button>
    <button onclick="fillExample()">Fill Example</button>
  </div>

  <h3>Request JSON</h3>
  <pre id="jsonPreview"></pre>

  <h3>Result</h3>
  <pre id="result"></pre>

  <script>
    const expected = {EXPECTED_COLUMNS};
    const numericFields = new Set({list(NUMERIC_FIELDS)});

    function buildPayload() {{
      const obj = {{}};
      for (const key of expected) {{
        const el = document.getElementById(key);
        if (!el) continue;
        let val = el.value;

        if (val === "") {{
          obj[key] = "";
          continue;
        }}

        if (numericFields.has(key)) {{
          const n = Number(val);
          obj[key] = isNaN(n) ? val : n;
        }} else {{
          obj[key] = val;
        }}
      }}
      return obj;
    }}

    function updatePreview() {{
      document.getElementById("jsonPreview").textContent = JSON.stringify(buildPayload(), null, 2);
    }}

    async function predictSingle() {{
      const resultEl = document.getElementById("result");
      resultEl.textContent = "Running...";
      updatePreview();

      try {{
        const payload = buildPayload();

        const res = await fetch("/predict", {{
          method: "POST",
          headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify(payload)
        }});

        const text = await res.text();
        resultEl.textContent = text;
      }} catch (e) {{
        resultEl.textContent = "Error: " + e.message;
      }}
    }}

    function clearAll() {{
      for (const key of expected) {{
        const el = document.getElementById(key);
        if (el) el.value = "";
      }}
      updatePreview();
      document.getElementById("result").textContent = "";
    }}

    function fillExample() {{
      const example = {{
        "shipment_id": "SHP_0001",
        "carrier": "DHL",
        "shipping_mode": "Air",
        "region": "APAC",
        "origin_country": "IN",
        "destination_country": "SG",
        "shipment_weight_kg": 10,
        "shipment_volume_cbm": 0.2,
        "priority_flag": 1,
        "fragile_flag": 0,
        "temperature_control_flag": 0,
        "planned_delivery_days": 5,
        "actual_delivery_days": 6,
        "delivery_delay_days": 1,
        "shipping_cost_usd": 120,
        "fuel_surcharge_pct": 8,
        "customs_delay_flag": 0,
        "weather_disruption_flag": 0,
        "shipment_value_usd": 500,
        "insurance_flag": 1
      }};

      for (const k in example) {{
        const el = document.getElementById(k);
        if (el) el.value = example[k];
      }}
      updatePreview();
    }}

    document.addEventListener("input", updatePreview);
    updatePreview();
  </script>
</body>
</html>
"""


# ---------------------------
# Single prediction
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON. Send a JSON object."}), 400

    missing = [c for c in EXPECTED_COLUMNS if c not in data or data[c] == ""]
    if missing:
        return jsonify({"error": "Missing required fields", "missing": missing}), 400

    df = pd.DataFrame([data])

    # Try primary model
    try:
        prob = float(xgb_model.predict_proba(df)[0][1])
        model_used = "XGBoost"
    except Exception:
        # Fallback
        prob = float(dt_model.predict_proba(df)[0][1])
        model_used = "DecisionTree"

    bucket, action = risk_bucket(prob)

    return jsonify({
        "model_used": model_used,
        "sla_breach_probability": round(prob, 3),
        "risk_bucket": bucket,
        "business_action": action
    })


# ---------------------------
# Batch prediction
# ---------------------------
@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)

    if not isinstance(data, list) or len(data) == 0:
        return jsonify({"error": "Invalid JSON. Send a list of objects."}), 400

    df = pd.DataFrame(data)

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
            "action": action
        })

    return jsonify(results)


# ---------------------------
# Entrypoint (Render + local)
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
