import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import joblib
import os

os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

joblib.dump(model, 'models/iris_model.joblib')
X_train.to_csv('data/baseline_data.csv', index=False)

print("Setup Complete: Model and Baseline data saved.")