import time
import joblib
import pandas as pd
from flask import Flask, render_template, request
from sdk.exporter import MLExporter, LATENCY, RESPONSE_TIME, PRED_PROBABILITY

app = Flask(__name__)

# --- Load Loan Eligibility Model ---
# Note: Ensure you have run your model_setup.py to generate this file
try:
    model = joblib.load('models/loan_model.joblib')
    print("[SUCCESS] Loan Eligibility Model loaded.")
except Exception as e:
    print(f"[ERROR] Loading model failed: {e}")

# Initialize Monitoring SDK with the new Loan Baseline
monitor = MLExporter(port=8000, baseline_path='data/baseline_data.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    start_api_time = time.time()
    monitor.track_system_health()
    
    try:
        # Extract 3 features from the simplified form
        # f1: Income, f2: LoanAmount, f3: CreditHistory
        input_features = [
            float(request.form['f1']), 
            float(request.form['f2']), 
            float(request.form['f3'])
        ]
        
        # Create DataFrame with matching column names from training
        df = pd.DataFrame([input_features], columns=['Income', 'LoanAmount', 'CreditHistory'])
        
        # Prediction and Internal Latency tracking
        with LATENCY.time():
            prediction = int(model.predict(df)[0])
            prob = float(model.predict_proba(df).max())
            PRED_PROBABILITY.observe(prob)

        # Log ML Metrics (Drift based on Loan Amount & Feature Ranges)
        # We pass the whole df, exporter will pick the relevant column for drift
        monitor.check_drift_and_features(df)
        
        # Performance Evaluation Loop (Calculates Precision, Recall, F1)
        actual = request.form.get('actual')
        if actual:
            monitor.track_performance_metrics(prediction, int(actual))

        # Capture total operational response time
        RESPONSE_TIME.observe(time.time() - start_api_time)
        
        result_text = "Loan APPROVED" if prediction == 1 else "Loan REJECTED"
        return render_template('index.html', 
                               prediction=result_text, 
                               probability=f"{prob*100:.2f}%")

    except Exception as e:
        monitor.log_error()
        print(f"Prediction Error: {e}")
        return render_template('index.html', error="Prediction failed. Please check inputs.")

if __name__ == "__main__":
    
    # Clean console summary of all active service endpoints
    print("\n" + "="*50)
    print("      LOAN ELIGIBILITY MONITORING SYSTEM")
    print("="*50)
    print(f"[*] Flask Web UI:     http://localhost:5000")
    print(f"[*] Metrics SDK:      http://localhost:8000/metrics")
    print(f"[*] Prometheus UI:    http://localhost:9090")
    print(f"[*] Grafana Dash:     http://localhost:3000")
    print("="*50)
    print("[INFO] Monitoring Stage 1: Internal SDK is active.")
    print("[INFO] Press CTRL+C to stop all services.\n")

    app.run(host='0.0.0.0', port=5000)