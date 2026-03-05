from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

from business_logic import risk_bucket

app = Flask(__name__)

# ---------------------------
# Load models safely
# ---------------------------
# NOTE: Update these paths based on where your .pkl files are.
# If your repo has a folder named "models" and the pkl files are inside it, keep as is.
# If your pkl files are in the repo root, change to: "xgb_model.pkl" and "dt_model.pkl"
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
# Home route (professional)
# ---------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "api": "Carrier SLA Risk Prediction API",
        "status": "running",
        "endpoints": {
            "single_prediction": "/predict",
            "batch_prediction": "/predict/batch"
        }
    })


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
