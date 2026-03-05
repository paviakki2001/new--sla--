from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import html

from business_logic import risk_bucket

app = Flask(__name__)

# =====================================================
# LOAD MODELS (ROOT FOLDER)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

XGB_MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.pkl")
DT_MODEL_PATH  = os.path.join(BASE_DIR, "dt_model.pkl")

xgb_model, dt_model = None, None

try:
    xgb_model = joblib.load(XGB_MODEL_PATH)
    print("✅ XGBoost model loaded")
except Exception as e:
    print("❌ XGBoost model load error:", e)

try:
    dt_model = joblib.load(DT_MODEL_PATH)
    print("✅ DecisionTree model loaded")
except Exception as e:
    print("❌ DecisionTree model load error:", e)

if xgb_model is None and dt_model is None:
    raise RuntimeError("No models loaded. Ensure xgb_model.pkl and dt_model.pkl exist in repo root.")

# =====================================================
# EXPECTED COLS (FROM MODEL IF POSSIBLE)
# =====================================================
EXPECTED_COLS = None

if xgb_model is not None and hasattr(xgb_model, "feature_names_in_"):
    EXPECTED_COLS = list(xgb_model.feature_names_in_)
elif dt_model is not None and hasattr(dt_model, "feature_names_in_"):
    EXPECTED_COLS = list(dt_model.feature_names_in_)

# fallback only if models don't provide feature names
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
# INPUT HELPERS
# =====================================================
NUMERIC_HINTS = ("_flag", "_days", "_pct", "_kg", "_cbm", "_usd")

def is_numeric_feature(col: str) -> bool:
    c = col.lower()
    return any(c.endswith(s) for s in NUMERIC_HINTS)

def coerce_value(col: str, val):
    if val is None:
        val = ""
    val = str(val).strip()

    if is_numeric_feature(col):
        if val == "":
            return 0
        try:
            return float(val)
        except:
            return 0
    return val

def sanitize_row(data: dict):
    row = {}
    for col in EXPECTED_COLS:
        row[col] = coerce_value(col, data.get(col))
    return row

def align_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = 0 if is_numeric_feature(col) else ""
    df = df.reindex(columns=EXPECTED_COLS, fill_value=0)
    return df

def get_probability(df: pd.DataFrame):
    if xgb_model is not None:
        try:
            p = float(xgb_model.predict_proba(df)[0][1])
            return p, "XGBoost"
        except Exception as e:
            print("⚠️ XGB predict failed, fallback DT:", e)

    if dt_model is not None:
        p = float(dt_model.predict_proba(df)[0][1])
        return p, "DecisionTree"

    raise RuntimeError("No model available for prediction.")

def build_explanation(row: dict):
    reasons = []

    delay = float(row.get("delivery_delay_days", 0) or 0)
    customs = float(row.get("customs_delay_flag", 0) or 0)
    weather = float(row.get("weather_disruption_flag", 0) or 0)
    actual = float(row.get("actual_delivery_days", 0) or 0)
    planned = float(row.get("planned_delivery_days", 0) or 0)
    fuel = float(row.get("fuel_surcharge_pct", 0) or 0)
    priority = float(row.get("priority_flag", 0) or 0)

    if delay >= 3:
        reasons.append(f"Delivery delay is high ({delay} days), which increases SLA breach risk.")
    elif delay > 0:
        reasons.append(f"Delivery delay exists ({delay} days), which may increase risk.")
    else:
        reasons.append("No delivery delay detected, which supports low SLA breach risk.")

    if planned > 0 and actual > planned:
        reasons.append(f"Actual delivery days ({actual}) exceeded planned days ({planned}).")

    if customs == 1:
        reasons.append("Customs delay flag is ON, which can increase breach risk.")

    if weather == 1:
        reasons.append("Weather disruption flag is ON, which can cause delays.")

    if fuel >= 15:
        reasons.append(f"Fuel surcharge is high ({fuel}%), which may indicate operational disruption.")

    if priority == 1:
        reasons.append("Priority shipment may reduce risk due to faster handling.")

    return reasons

# Debug storage (optional)
LAST_INPUT = {"received": None, "aligned_columns": None}

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

@app.route("/debug/last-input", methods=["GET"])
def debug_last_input():
    return jsonify(LAST_INPUT)

# =====================================================
# PROFESSIONAL UI PAGE (NO RAW JSON ON UI)
# =====================================================
@app.route("/ui", methods=["GET"])
def ui():
    inputs_html = []

    for col in EXPECTED_COLS:
        input_type = "number" if is_numeric_feature(col) else "text"
        step_attr = ' step="any"' if input_type == "number" else ""
        placeholder = "0" if input_type == "number" else f"Enter {col}"

        inputs_html.append(f"""
        <div class="row">
          <label for="{html.escape(col)}">{html.escape(col)}</label>
          <input id="{html.escape(col)}" name="{html.escape(col)}" type="{input_type}"{step_attr}
                 placeholder="{html.escape(placeholder)}" />
        </div>
        """)

    page = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Carrier SLA Risk Prediction</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      max-width: 980px;
      margin: 20px auto;
      padding: 0 16px;
      background: #f7f7f7;
    }}
    .card {{
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 16px;
      background: #fff;
    }}
    h2 {{
      margin: 0 0 6px 0;
    }}
    .small {{
      font-size: 12px;
      color: #555;
    }}
    .row {{
      display: grid;
      grid-template-columns: 330px 1fr;
      gap: 12px;
      align-items: center;
      padding: 6px 0;
    }}
    label {{
      font-weight: 600;
      color: #222;
    }}
    input {{
      padding: 9px;
      border: 1px solid #ccc;
      border-radius: 10px;
      width: 100%;
      outline: none;
    }}
    input:focus {{
      border-color: #333;
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
      border-radius: 10px;
      cursor: pointer;
    }}
    button.secondary {{
      background: #fff;
      color: #111;
    }}
    .result-box {{
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 14px;
      background: #fcfcfc;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 10px;
    }}
    .kv {{
      border: 1px solid #eee;
      padding: 12px;
      border-radius: 12px;
      background: #fff;
    }}
    .kv .k {{
      font-size: 12px;
      color: #666;
      margin-bottom: 6px;
    }}
    .kv .v {{
      font-size: 18px;
      font-weight: 700;
      color: #111;
    }}
    .badge {{
      display: inline-block;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid #ddd;
      background: #fff;
    }}
    .badge.low {{ border-color: #bfe3bf; }}
    .badge.medium {{ border-color: #ffe2a8; }}
    .badge.high {{ border-color: #ffb3b3; }}
    ul {{
      margin: 10px 0 0 18px;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h2>Carrier SLA Risk Prediction</h2>
    <div class="small">
      Enter shipment details and click <b>Predict</b>. Empty numeric fields will be treated as 0.
    </div>
  </div>

  <div class="card">
    <form id="predForm">
      {''.join(inputs_html)}
      <div class="actions">
        <button type="button" onclick="submitPredict()">Predict</button>
        <button type="button" class="secondary" onclick="resetForm()">Clear</button>
      </div>
    </form>
  </div>

  <div class="card">
    <h3>Prediction Result</h3>
    <div id="result" class="result-box">No prediction yet.</div>
  </div>

<script>
function resetForm() {{
  document.getElementById('predForm').reset();
  document.getElementById('result').innerHTML = "No prediction yet.";
}}

function fmtPct(x) {{
  try {{
    const v = parseFloat(x);
    return (v * 100).toFixed(1) + "%";
  }} catch(e) {{
    return x;
  }}
}}

function riskClass(bucket) {{
  const b = (bucket || "").toLowerCase();
  if (b.includes("high")) return "high";
  if (b.includes("med")) return "medium";
  return "low";
}}

async function submitPredict() {{
  const resultEl = document.getElementById("result");
  resultEl.innerHTML = "Running prediction...";

  const form = document.getElementById("predForm");
  const data = new FormData(form);

  const payload = {{}};
  for (const pair of data.entries()) {{
    payload[pair[0]] = pair[1];
  }}

  try {{
    const res = await fetch("/ui/predict", {{
      method: "POST",
      headers: {{"Content-Type":"application/json"}},
      body: JSON.stringify(payload)
    }});

    const out = await res.json();

    if (!res.ok) {{
      resultEl.innerHTML = "<b>Error:</b> " + (out.error || "Unknown error");
      return;
    }}

    let expHtml = "";
    if (out.explanations && out.explanations.length > 0) {{
      expHtml = "<ul>" + out.explanations.map(function(x) {{
        return "<li>" + x + "</li>";
      }}).join("") + "</ul>";
    }} else {{
      expHtml = "<div class='small'>No explanation available.</div>";
    }}

    const cls = riskClass(out.risk_bucket);

    resultEl.innerHTML =
      "<div class='grid'>" +
        "<div class='kv'><div class='k'>Model Used</div><div class='v'>" + out.model_used + "</div></div>" +
        "<div class='kv'><div class='k'>SLA Breach Probability</div><div class='v'>" + fmtPct(out.sla_breach_probability) + "</div></div>" +
        "<div class='kv'><div class='k'>Risk Bucket</div><div class='v'><span class='badge " + cls + "'>" + out.risk_bucket + "</span></div></div>" +
        "<div class='kv'><div class='k'>Recommended Action</div><div class='v'>" + out.business_action + "</div></div>" +
      "</div>" +
      "<h4 style='margin-top:14px;'>Why this result?</h4>" +
      expHtml;

  }} catch (e) {{
    resultEl.innerHTML = "<b>Error:</b> " + e.message;
  }}
}}
</script>
</body>
</html>
"""
    return page

# =====================================================
# UI PREDICT ENDPOINT (UI uses this)
# =====================================================
@app.route("/ui/predict", methods=["POST"])
def ui_predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid input. Please fill the form and try again."}), 400

    row = sanitize_row(data)
    df = pd.DataFrame([row])
    df = align_df(df)

    LAST_INPUT["received"] = row
    LAST_INPUT["aligned_columns"] = list(df.columns)

    prob, model_used = get_probability(df)
    bucket, action = risk_bucket(prob)
    explanations = build_explanation(row)

    return jsonify({
        "model_used": model_used,
        "sla_breach_probability": round(prob, 4),
        "risk_bucket": bucket,
        "business_action": action,
        "explanations": explanations
    })

# =====================================================
# JSON API ENDPOINTS (still available)
# =====================================================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON. Send a JSON object."}), 400

    row = sanitize_row(data)
    df = pd.DataFrame([row])
    df = align_df(df)

    prob, model_used = get_probability(df)
    bucket, action = risk_bucket(prob)

    return jsonify({
        "model_used": model_used,
        "sla_breach_probability": round(prob, 4),
        "risk_bucket": bucket,
        "business_action": action,
        "explanations": build_explanation(row)
    })

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)
    if not isinstance(data, list) or len(data) == 0:
        return jsonify({"error": "Invalid JSON. Send a list of objects."}), 400

    df = pd.DataFrame(data)
    df = align_df(df)

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
    for i, p in enumerate(probs):
        p = float(p)
        bucket, action = risk_bucket(p)
        row_dict = df.iloc[i].to_dict()
        results.append({
            "model_used": model_used,
            "sla_breach_probability": round(p, 4),
            "risk_bucket": bucket,
            "business_action": action,
            "explanations": build_explanation(row_dict)
        })

    return jsonify(results)

# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
