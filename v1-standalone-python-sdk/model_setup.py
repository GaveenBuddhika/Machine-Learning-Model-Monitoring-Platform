import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 1. Generate Synthetic Loan Dataset (Simple & Practical)
# Features: Income, LoanAmount, CreditHistory (1: Good, 0: Bad)
np.random.seed(42)
data_size = 1000

income = np.random.randint(2000, 10000, data_size)
loan_amount = np.random.randint(1000, 5000, data_size)
credit_history = np.random.choice([0, 1], data_size, p=[0.3, 0.7])

# Target: Approved (1) if Credit History is Good AND Income is sufficient
# Logic: If credit_history == 1 and income > (loan_amount * 1.5)
target = ((credit_history == 1) & (income > (loan_amount * 1.2))).astype(int)

df = pd.DataFrame({
    'Income': income,
    'LoanAmount': loan_amount,
    'CreditHistory': credit_history,
    'Outcome': target
})

# 2. Split Features and Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. Save Baseline Data (for drift detection)
# We save the features for the SDK to compare distribution shifts
X.to_csv('data/baseline_data.csv', index=False)
print("[SUCCESS] Baseline data saved to data/baseline_data.csv")

# 4. Train a simple Random Forest Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 5. Save the trained model
joblib.dump(model, 'models/loan_model.joblib')
print("[SUCCESS] Loan Model saved to models/loan_model.joblib")