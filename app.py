from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None

    if request.method == 'POST':
        try:
            # Read input from form
            inputs = [
                float(request.form['pe_ratio']),
                float(request.form['ps_ratio']),
                float(request.form['pb_ratio']),
                float(request.form['debt_to_equity']),
                float(request.form['free_cash_flow_yield']),
                float(request.form['price_momentum_3m']),
                float(request.form['news_sentiment_score'])
            ]

            # Scale input and predict
            features_scaled = scaler.transform([inputs])
            pred = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0][1]

            prediction = "BUY" if pred == 1 else "AVOID"
            confidence = f"{proba * 100:.2f}%"
        except Exception as e:
            prediction = "Error: Invalid input"
            confidence = str(e)

    return render_template("index.html", prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
git commit -m "Removed invalid git command"
git push origin main
