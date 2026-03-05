from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

from business_logic import risk_bucket

app = Flask(__name__)

# ---------------------------
# Model paths (choose ONE)
# ---------------------------
# If your .pkl files are in the SAME folder as app.py:
XGB_MODEL_PATH = "xgb_model.pkl"
DT_MODEL_PATH = "dt_model.pkl"

# If your .pkl files are inside models/ folder, use this instead:
# XGB_MODEL_PATH = "models/xgb_model.pkl"
# DT_MODEL_PATH = "models/dt_model.pkl"

xgb_model = joblib.load(XGB_MODEL_PATH)
dt_model = joblib.load(DT_MODEL_PATH)

# These must match your model expected columns
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

# Define which columns are numeric vs text
NUMERIC_FIELDS = {
    "shipment_weight_kg", "shipment_volume_cbm",
    "priority_flag", "fragile_flag", "temperature_control_flag",
    "planned_delivery_days", "actual_delivery_days", "delivery_delay_days",
    "shipping_cost_usd", "fuel_surcharge_pct",
    "customs_delay_flag", "weather_disruption_flag",
    "shipment_value_usd", "insurance_flag"
}

TEXT_FIELDS = [c for c in EXPECTED_COLUMNS if c not in NUMERIC_FIELDS]


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


@app.route("/ui", methods=["GET"])
def ui():
    # Simple form-based UI that builds JSON automatically
    # No sample hardcoding; user can enter ANY values.
    inputs_html = ""

    # text fields
    for f in TEXT_FIELDS:
        inputs_html += f"""
        <tr>
          <td style="padding:6px;"><b>{f}</b></td>
          <td style="padding:6px;"><input type="text" id="{f}" style="width: 100%;" placeholder="Enter {f}"></td>
        </tr>
        """

    # numeric fields
    for f in sorted(NUMERIC_FIELDS):
        inputs_html += f"""
        <tr>
          <td style="padding:6px;"><b>{f}</b></td>
          <td style="padding:6px;"><input type="number" step="any" id="{f}" style="width: 100%;" placeholder="Enter {f}"></td>
        </tr>
        """

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Carrier SLA Risk - Simple UI</title>
</head>
<body style="font-family: Arial, sans-serif; max-width: 980px; margin: 20px;">
  <h2>Carrier SLA Risk Prediction (Simple UI)</h2>
  <p>Fill the fields and click Predict. This works for any numbers you enter.</p>

  <table style="width:100%; border-collapse: collapse;">
    {inputs_html}
  </table>

  <br/>
  <button onclick="predictSingle()" style="padding:10px 14px;">Predict</button>
  <button onclick="fillExample()" style="padding:10px 14px; margin-left:8px;">Fill Example</button>

  <hr/>
  <h3>Request JSON</h3>
  <pre id="jsonPreview" style="background:#f6f6f6; padding:10px; border:1px solid #ddd;"></pre>

  <h3>Result</h3>
  <pre id="result" style="background:#f6f6f6; padding:10px; border:1px solid #ddd; white-space: pre-wrap;"></pre>

  <script>
    const numericFields = new Set({list(NUMERIC_FIELDS)});
    const expected = {EXPECTED_COLUMNS};

    function buildPayload() {{
      const obj = {{}};
      for (const key of expected) {{
        const el = document.getElementById(key);
        if (!el) continue;
        let val = el.value;

        // Keep empty as empty string (so user sees missing fields)
        if (val === "") {{
          obj[key] = "";
          continue;
        }}

        // Convert numeric fields to numbers
        if (numericFields.has(key)) {{
          const num = Number(val);
          obj[key] = isNaN(num) ? val : num;
        }} else {{
          obj[key] = val;
        }}
      }}
      return obj;
    }}

    function updatePreview() {{
      const payload = buildPayload();
      document.getElementById("jsonPreview").textContent = JSON.stringify(payload, null, 2);
    }}

    async function predictSingle() {{
      const resultEl = document.getElementById("result");
      resultEl.textContent = "Running...";
      try {{
        const payload = buildPayload();
        updatePreview();

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

    // Optional helper to quickly fill a working example
    function fillExample() {{
      const example = {{
        "shipment_id":"SHP_0001",
        "carrier":"DHL",
        "shipping_mode":"Air",
        "region":"APAC",
        "origin_country":"IN",
        "destination_country":"SG",
        "shipment_weight_kg":10,
        "shipment_volume_cbm":0.2,
        "priority_flag":1,
        "fragile_flag":0,
        "temperature_control_flag":0,
        "planned_delivery_days":5,
        "actual_delivery_days":6,
        "delivery_delay_days":1,
        "shipping_cost_usd":120,
        "fuel_surcharge_pct":8,
        "customs_delay_flag":0,
        "weather_disruption_flag":0,
        "shipment_value_usd":500,
        "insurance_flag":1
      }};

      for (const k in example) {{
        const el = document.getElementById(k);
        if (el) el.value = example[k];
      }}
      updatePreview();
    }}

    // Auto-update JSON preview when user types
    document.addEventListener("input", updatePreview);
    updatePreview();
  </script>
</body>
</html>
"""


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON. Send a JSON object."}), 400

    missing = [c for c in EXPECTED_COLUMNS if c not in data or data[c] == ""]
    if missing:
        return jsonify({"error": "Missing required fields", "missing": missing}), 400

    df = pd.DataFrame([data])

    try:
        prob = float(xgb_model.predict_proba(df)[0][1])
        model_used = "XGBoost"
    except Exception:
        prob = float(dt_model.predict_proba(df)[0][1])
        model_used = "DecisionTree"

    bucket, action = risk_bucket(prob)

    return jsonify({
        "model_used": model_used,
        "sla_breach_probability": round(prob, 3),
        "risk_bucket": bucket,
        "business_action": action
    })


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
