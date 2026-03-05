from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

from business_logic import risk_bucket

app = Flask(__name__)

# ---------------------------
# Load models safely
# ---------------------------
# Your models are in repo root as per your setting:
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


# ---------------------------
# Home route (simple + professional)
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
# Simple UI route
# ---------------------------
@app.route("/ui", methods=["GET"])
def ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Carrier SLA Risk API - Simple UI</title>
</head>
<body style="font-family: Arial, sans-serif; max-width: 900px; margin: 20px;">
  <h2>Carrier SLA Risk Prediction (Simple UI)</h2>

  <p><b>Single Prediction</b> (JSON object)</p>
  <textarea id="singleInput" rows="10" style="width: 100%;"></textarea><br/>
  <button onclick="singlePredict()">Predict (Single)</button>

  <hr/>

  <p><b>Batch Prediction</b> (JSON array of objects)</p>
  <textarea id="batchInput" rows="10" style="width: 100%;"></textarea><br/>
  <button onclick="batchPredict()">Predict (Batch)</button>

  <hr/>

  <p><b>Result</b></p>
  <pre id="result" style="background:#f6f6f6; padding:10px; border:1px solid #ddd; white-space: pre-wrap;"></pre>

  <script>
    async function singlePredict() {
      const resultEl = document.getElementById("result");
      resultEl.textContent = "Running...";
      try {
        const payload = JSON.parse(document.getElementById("singleInput").value);
        const res = await fetch("/predict", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const text = await res.text();
        resultEl.textContent = text;
      } catch (e) {
        resultEl.textContent = "Error: " + e.message;
      }
    }

    async function batchPredict() {
      const resultEl = document.getElementById("result");
      resultEl.textContent = "Running...";
      try {
        const payload = JSON.parse(document.getElementById("batchInput").value);
        const res = await fetch("/predict/batch", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const text = await res.text();
        resultEl.textContent = text;
      } catch (e) {
        resultEl.textContent = "Error: " + e.message;
      }
    }

    // Optional: Paste your real feature JSON here for quick testing.
    // If you already have single.json and batch.json in your repo, copy their content into these boxes.
    document.getElementById("singleInput").value = JSON.stringify({
      "paste_your_feature_1": 0,
      "paste_your_feature_2": 0
    }, null, 2);

    document.getElementById("batchInput").value = JSON.stringify([
      {
        "paste_your_feature_1": 0,
        "paste_your_feature_2": 0
      },
      {
        "paste_your_feature_1": 0,
        "paste_your_feature_2": 0
      }
    ], null, 2);
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
        return jsonify({"error": "Invalid JSON. Send a JSON object with feature keys."}), 400

    df = pd.DataFrame([data])

    # Try primary model first
    try:
        prob = float(xgb_model.predict_proba(df)[0][1])
        model_used = "XGBoost"
    except Exception:
        # Fallback model
        try:
            prob = float(dt_model.predict_proba(df)[0][1])
            model_used = "DecisionTree"
        except Exception as e:
            return jsonify({
                "error": "Prediction failed for both models.",
                "details": str(e)
            }), 500

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
        return jsonify({"error": "Invalid JSON. Send a JSON array (list) of feature objects."}), 400

    df = pd.DataFrame(data)

    try:
        probs = xgb_model.predict_proba(df)[:, 1]
    except Exception:
        try:
            probs = dt_model.predict_proba(df)[:, 1]
        except Exception as e:
            return jsonify({
                "error": "Batch prediction failed for both models.",
                "details": str(e)
            }), 500

    results = []
    for p in probs:
        p = float(p)
        bucket, action = risk_bucket(p)
        results.append({
            "sla_breach_probability": round(p, 3),
            "risk_bucket": bucket,
            "action": action
        })

    return jsonify(results)


# ---------------------------
# Render entrypoint
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

