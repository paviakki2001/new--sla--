from flask import Flask, request, jsonify
import pandas as pd
import joblib

from business_logic import risk_bucket

app = Flask(__name__)

# Load models
xgb_model = joblib.load("models/xgb_model.pkl")
print("MODEL TYPE:", type(xgb_model))

if hasattr(xgb_model, "feature_names_in_"):
    print("Expected input columns:", list(xgb_model.feature_names_in_))

if hasattr(xgb_model, "named_steps"):
    for name, step in xgb_model.named_steps.items():
        print("Pipeline step:", name, "->", type(step))
        if hasattr(step, "feature_names_in_"):
            print("Step expected columns:", list(step.feature_names_in_))
dt_model = joblib.load("models/dt_model.pkl")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json
        df = pd.DataFrame([data])

        # Primary model (XGBoost)
        prob = xgb_model.predict_proba(df)[0][1]

        bucket, action = risk_bucket(prob)

        return jsonify({
            "model_used": "XGBoost",
            "sla_breach_probability": round(float(prob),3),
            "risk_bucket": bucket,
            "business_action": action
        })

    except:

        # Fallback model
        df = pd.DataFrame([data])

        prob = dt_model.predict_proba(df)[0][1]

        bucket, action = risk_bucket(prob)

        return jsonify({
            "model_used": "DecisionTree",
            "sla_breach_probability": round(float(prob),3),
            "risk_bucket": bucket,
            "business_action": action
        })


@app.route("/predict/batch", methods=["POST"])
def predict_batch():

    data = request.json
    df = pd.DataFrame(data)

    probs = xgb_model.predict_proba(df)[:,1]

    results = []

    for p in probs:

        bucket, action = risk_bucket(p)

        results.append({
            "sla_breach_probability": round(float(p),3),
            "risk_bucket": bucket,
            "action": action
        })

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)

    print("MODEL TYPE:", type(xgb_model))

# If the model is a sklearn Pipeline with preprocessing
if hasattr(xgb_model, "feature_names_in_"):
    print("Expected input columns:", list(xgb_model.feature_names_in_))

# If it's a Pipeline containing a ColumnTransformer
if hasattr(xgb_model, "named_steps"):
    for name, step in xgb_model.named_steps.items():
        print("Step:", name, "Type:", type(step))
        if hasattr(step, "feature_names_in_"):
            print("Step expected columns:", list(step.feature_names_in_))  
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)