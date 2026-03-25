import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# 加载模型
model = joblib.load('sentiment_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in JSON payload'}), 400

        text = data['text']
        proba = model.predict_proba([text])[0]
        pred_class = proba.argmax()
        confidence = float(proba.max())
        sentiment = "positive" if pred_class == 1 else "negative"

        return jsonify({
            'sentiment': sentiment,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
