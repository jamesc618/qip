import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Example training data
data = pd.DataFrame({
    'pe_ratio': [10, 25, 15, 40, 8],
    'pb_ratio': [1.0, 5.0, 1.5, 6.0, 0.8],
    'de_ratio': [0.3, 2.0, 0.5, 3.0, 0.2],
    'market_cap': [500, 10000, 750, 20000, 400],
    'revenue_growth': [15, -5, 10, -10, 20],
    'label': [1, 0, 1, 0, 1]
})

X = data.drop('label', axis=1)
y = data['label']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'model/model.pkl')
