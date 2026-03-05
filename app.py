from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

from business_logic import risk_bucket

app = Flask(__name__)

# ---------------------------
# Load models safely
# ---------------------------
# If your .pkl files are in repo root, keep as below.
# If they are in models/ folder, change to: "models/xgb_model.pkl" and "models/dt_model.pkl"
XGB_MODEL_PATH = "xgb_model.pkl"
DT_MODEL_PATH = "dt_model.pkl"

try:
    xgb_model = joblib.load(XGB_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load XGBoost model from {XGB_MODEL_PATH}: {e}")

try:
    dt_model = joblib.load(DT_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load DecisionTree model from {DT_MODEL_PATH}: {e}")

# Expected feature list (based on what your pipeline printed)
EXPECTED_FEATURES = [
    "shipment_id",
    "carrier",
    "shipping_mode",
    "region",
    "origin_country",
    "destination_country",
    "shipment_weight_kg",
    "shipment_volume_cbm",
    "priority_flag",
    "fragile_flag",
    "temperature_control_flag",
    "planned_delivery_days",
    "actual_delivery_days",
    "delivery_delay_days",
    "shipping_cost_usd",
    "fuel_surcharge_pct",
    "customs_delay_flag",
    "weather_disruption_flag",
    "shipment_value_usd",
    "insurance_flag",
]

# Basic options for UI dropdowns (you can edit these anytime)
CARRIER_OPTIONS = ["DHL", "FedEx", "UPS", "BlueDart", "DTDC", "Delhivery", "Other"]
SHIPPING_MODE_OPTIONS = ["Air", "Sea", "Road", "Rail"]
REGION_OPTIONS = ["APAC", "EMEA", "AMER", "India", "Other"]
COUNTRY_OPTIONS = ["IN", "SG", "AE", "US", "UK", "DE", "FR", "CN", "JP", "Other"]


# ---------------------------
# Helpers
# ---------------------------
def _to_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def _to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def validate_payload(payload: dict):
    """Returns (ok: bool, message: str)."""
    missing = [k for k in EXPECTED_FEATURES if k not in payload]
    if missing:
        return False, f"Missing required fields: {missing}"
    return True, ""


def predict_probability(df: pd.DataFrame):
    """Try XGB first, then fallback to DT. Returns (prob, model_used)."""
    try:
        prob = float(xgb_model.predict_proba(df)[0][1])
        return prob, "XGBoost"
    except Exception:
        prob = float(dt_model.predict_proba(df)[0][1])
        return prob, "DecisionTree"


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
# Usable Simple UI (HTML Form)
# ---------------------------
@app.route("/ui", methods=["GET"])
def ui():
    # Simple, functional UI with a form. Submits JSON to /predict.
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Carrier SLA Risk Prediction - UI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
</head>
<body style="font-family: Arial, sans-serif; max-width: 980px; margin: 18px;">
  <h2>Carrier SLA Risk Prediction (Simple UI)</h2>
  <p style="color:#555;margin-top:-8px;">
    Fill inputs → click <b>Predict</b>. Output appears below.
  </p>

  <div style="padding:12px;border:1px solid #ddd;border-radius:8px;">
    <h3 style="margin-top:0;">Shipment Details</h3>

    <label>Shipment ID</label><br/>
    <input id="shipment_id" style="width:100%; padding:8px;" value="SHP_0001"/><br/><br/>

    <label>Carrier</label><br/>
    <select id="carrier" style="width:100%; padding:8px;">
      {''.join([f'<option value="{c}">{c}</option>' for c in CARRIER_OPTIONS])}
    </select><br/><br/>

    <label>Shipping Mode</label><br/>
    <select id="shipping_mode" style="width:100%; padding:8px;">
      {''.join([f'<option value="{m}">{m}</option>' for m in SHIPPING_MODE_OPTIONS])}
    </select><br/><br/>

    <label>Region</label><br/>
    <select id="region" style="width:100%; padding:8px;">
      {''.join([f'<option value="{r}">{r}</option>' for r in REGION_OPTIONS])}
    </select><br/><br/>

    <label>Origin Country</label><br/>
    <select id="origin_country" style="width:100%; padding:8px;">
      {''.join([f'<option value="{c}">{c}</option>' for c in COUNTRY_OPTIONS])}
    </select><br/><br/>

    <label>Destination Country</label><br/>
    <select id="destination_country" style="width:100%; padding:8px;">
      {''.join([f'<option value="{c}">{c}</option>' for c in COUNTRY_OPTIONS])}
    </select><br/><br/>

    <h3>Numeric Inputs</h3>

    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
      <div>
        <label>Shipment Weight (kg)</label><br/>
        <input id="shipment_weight_kg" type="number" step="0.01" style="width:100%; padding:8px;" value="10"/>
      </div>
      <div>
        <label>Shipment Volume (cbm)</label><br/>
        <input id="shipment_volume_cbm" type="number" step="0.01" style="width:100%; padding:8px;" value="0.2"/>
      </div>

      <div>
        <label>Planned Delivery Days</label><br/>
        <input id="planned_delivery_days" type="number" step="1" style="width:100%; padding:8px;" value="5"/>
      </div>
      <div>
        <label>Actual Delivery Days</label><br/>
        <input id="actual_delivery_days" type="number" step="1" style="width:100%; padding:8px;" value="6"/>
      </div>

      <div>
        <label>Delivery Delay Days</label><br/>
        <input id="delivery_delay_days" type="number" step="1" style="width:100%; padding:8px;" value="1"/>
      </div>
      <div>
        <label>Shipping Cost (USD)</label><br/>
        <input id="shipping_cost_usd" type="number" step="0.01" style="width:100%; padding:8px;" value="120"/>
      </div>

      <div>
        <label>Fuel Surcharge (%)</label><br/>
        <input id="fuel_surcharge_pct" type="number" step="0.01" style="width:100%; padding:8px;" value="8"/>
      </div>
      <div>
        <label>Shipment Value (USD)</label><br/>
        <input id="shipment_value_usd" type="number" step="0.01" style="width:100%; padding:8px;" value="500"/>
      </div>
    </div>

    <h3>Flags (0/1)</h3>
    <p style="color:#666;margin-top:-8px;">Tick = 1, Untick = 0</p>

    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px;">
      <label><input type="checkbox" id="priority_flag" checked/> Priority</label>
      <label><input type="checkbox" id="fragile_flag"/> Fragile</label>
      <label><input type="checkbox" id="temperature_control_flag"/> Temperature Control</label>
      <label><input type="checkbox" id="customs_delay_flag"/> Customs Delay</label>
      <label><input type="checkbox" id="weather_disruption_flag"/> Weather Disruption</label>
      <label><input type="checkbox" id="insurance_flag" checked/> Insurance</label>
    </div>

    <br/>
    <button onclick="doPredict()" style="padding:10px 16px; font-size: 15px;">Predict</button>
    <button onclick="fillSample()" style="padding:10px 16px; font-size: 15px; margin-left:8px;">Load Sample</button>
  </div>

  <h3>Result</h3>
  <pre id="result" style="background:#f6f6f6; padding:12px; border:1px solid #ddd; border-radius:8px; white-space: pre-wrap;"></pre>

  <h3>Payload (JSON)</h3>
  <pre id="payload" style="background:#fff; padding:12px; border:1px solid #ddd; border-radius:8px; white-space: pre-wrap;"></pre>

<script>
  function flag(id) {{
    return document.getElementById(id).checked ? 1 : 0;
  }}

  function val(id) {{
    return document.getElementById(id).value;
  }}

  function num(id) {{
    const v = parseFloat(document.getElementById(id).value);
    return isNaN(v) ? 0 : v;
  }}

  function intNum(id) {{
    const v = parseInt(document.getElementById(id).value);
    return isNaN(v) ? 0 : v;
  }}

  function buildPayload() {{
    // Must match the model's expected keys exactly
    return {{
      shipment_id: val("shipment_id"),
      carrier: val("carrier"),
      shipping_mode: val("shipping_mode"),
      region: val("region"),
      origin_country: val("origin_country"),
      destination_country: val("destination_country"),
      shipment_weight_kg: num("shipment_weight_kg"),
      shipment_volume_cbm: num("shipment_volume_cbm"),
      priority_flag: flag("priority_flag"),
      fragile_flag: flag("fragile_flag"),
      temperature_control_flag: flag("temperature_control_flag"),
      planned_delivery_days: intNum("planned_delivery_days"),
      actual_delivery_days: intNum("actual_delivery_days"),
      delivery_delay_days: intNum("delivery_delay_days"),
      shipping_cost_usd: num("shipping_cost_usd"),
      fuel_surcharge_pct: num("fuel_surcharge_pct"),
      customs_delay_flag: flag("customs_delay_flag"),
      weather_disruption_flag: flag("weather_disruption_flag"),
      shipment_value_usd: num("shipment_value_usd"),
      insurance_flag: flag("insurance_flag"),
    }};
  }}

  async function doPredict() {{
    const resultEl = document.getElementById("result");
    const payloadEl = document.getElementById("payload");

    resultEl.textContent = "Running...";
    const payload = buildPayload();
    payloadEl.textContent = JSON.stringify(payload, null, 2);

    try {{
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

  function fillSample() {{
    document.getElementById("shipment_id").value = "SHP_0002";
    document.getElementById("carrier").value = "FedEx";
    document.getElementById("shipping_mode").value = "Air";
    document.getElementById("region").value = "APAC";
    document.getElementById("origin_country").value = "IN";
    document.getElementById("destination_country").value = "SG";
    document.getElementById("shipment_weight_kg").value = 8;
    document.getElementById("shipment_volume_cbm").value = 0.15;
    document.getElementById("planned_delivery_days").value = 4;
    document.getElementById("actual_delivery_days").value = 6;
    document.getElementById("delivery_delay_days").value = 2;
    document.getElementById("shipping_cost_usd").value = 95;
    document.getElementById("fuel_surcharge_pct").value = 7.5;
    document.getElementById("shipment_value_usd").value = 400;

    document.getElementById("priority_flag").checked = true;
    document.getElementById("fragile_flag").checked = true;
    document.getElementById("temperature_control_flag").checked = false;
    document.getElementById("customs_delay_flag").checked = false;
    document.getElementById("weather_disruption_flag").checked = true;
    document.getElementById("insurance_flag").checked = true;
  }}

  // Load JSON preview initially
  document.getElementById("payload").textContent = JSON.stringify(buildPayload(), null, 2);
  document.getElementById("result").textContent = "Click Predict to run.";
</script>
</body>
</html>
"""


# ---------------------------
# Single prediction (API)
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON. Send a JSON object with feature keys."}), 400

    ok, msg = validate_payload(data)
    if not ok:
        return jsonify({"error": msg}), 400

    df = pd.DataFrame([data])

    try:
        prob, model_used = predict_probability(df)
    except Exception as e:
        return jsonify({"error": "Prediction failed.", "details": str(e)}), 500

    bucket, action = risk_bucket(prob)

    return jsonify({
        "model_used": model_used,
        "sla_breach_probability": round(prob, 3),
        "risk_bucket": bucket,
        "business_action": action
    })


# ---------------------------
# Batch prediction (API)
# ---------------------------
@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)

    if not isinstance(data, list) or len(data) == 0:
        return jsonify({"error": "Invalid JSON. Send a JSON array (list) of feature objects."}), 400

    # Validate each row
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            return jsonify({"error": f"Row {idx} must be a JSON object."}), 400
        ok, msg = validate_payload(row)
        if not ok:
            return jsonify({"error": f"Row {idx}: {msg}"}), 400

    df = pd.DataFrame(data)

    # Try XGB then DT for batch
    try:
        probs = xgb_model.predict_proba(df)[:, 1]
        model_used = "XGBoost"
    except Exception:
        try:
            probs = dt_model.predict_proba(df)[:, 1]
            model_used = "DecisionTree"
        except Exception as e:
            return jsonify({"error": "Batch prediction failed.", "details": str(e)}), 500

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
# Render / Local entrypoint
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug=False is better when testing UI (no auto-reload duplicate prints)
    app.run(host="0.0.0.0", port=port, debug=False)
