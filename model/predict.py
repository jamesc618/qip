import pandas as pd
import joblib

model = joblib.load('model/model.pkl')

def predict_from_input(features):
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    return "BUY" if prediction == 1 else "AVOID"

def predict_from_csv(file):
    df = pd.read_csv(file)
    prediction = model.predict(df)[0]
    return "BUY" if prediction == 1 else "AVOID"
