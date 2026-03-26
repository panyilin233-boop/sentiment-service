import os
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')
model = joblib.load(model_path)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Sentiment API is running",
        "endpoint": "/predict",
        "method": "POST"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not data or "text" not in data:
            return jsonify({"error": 'Missing "text" field in JSON payload'}), 400

        text = data["text"]
        proba = model.predict_proba([text])[0]
        pred_class = int(proba.argmax())
        confidence = float(proba.max())
        sentiment = "positive" if pred_class == 1 else "negative"

        return jsonify({
            "sentiment": sentiment,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
