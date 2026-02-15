# Universal ML Monitoring Framework: Stage 1 (Internal SDK)

This framework implements a **Native Instrumentation Pattern** to
provide real-time observability for Machine Learning models.

It monitors:

-   Infrastructure Health (CPU, RAM, Disk)
-   Operational Efficiency (Latency, Errors)
-   ML Quality (Accuracy, Precision, Recall, F1-Score)
-   Data Drift (KS Statistical Test)

------------------------------------------------------------------------

# 1. Project Structure

The following directory structure is **mandatory** to ensure:

-   Proper Docker volume mapping
-   Correct Python module resolution
-   Clean separation of concerns

```
    ├── app.py                  # Serving Layer (Flask App)
    ├── model_setup.py          # Provisioning Layer (Model Training)
    ├── exporter.py             # Monitoring SDK (Prometheus Exporter Logic)
    ├── Dockerfile              # Application Container Definition
    ├── requirements.txt        # Python Dependencies
    ├── docker-compose.yml      # Multi-Service Orchestration
    ├── data/                   # Baseline Data Storage
    │   └── baseline_data.csv
    ├── models/                 # Model Artifacts
    │   └── loan_model.joblib
    ├── templates/              # Frontend UI
    │   └── index.html
    └── grafana/
        └── provisioning/
            └── dashboards/
                ├── dashboard.yaml      # Dashboard provisioning config
                └── dashboard.json      # grafana dashboard template
            └── datasources/
                ├── datasource.yaml     # Prometheus datasource config

```
------------------------------------------------------------------------

# 2. Dockerfile (Application Container Layer)

Below is a recommended production-ready Dockerfile for the Flask +
Monitoring SDK application.

``` dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y     build-essential     && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Flask and Metrics ports
EXPOSE 5000
EXPOSE 8000

# Start Application
CMD ["python", "app.py"]
```

### Why This Dockerfile Structure?

-   Uses lightweight `python:3.10-slim`
-   Optimized layer caching
-   Clean working directory separation
-   Supports both Flask app (5000) and Prometheus exporter (8000)

------------------------------------------------------------------------

# 3. Model Migration Guide (Loan Model -> other model)

When switching to a new model, update **three key layers**.

------------------------------------------------------------------------

## A. model_setup.py (Provisioning Layer)

### Goal

Define feature set and export baseline dataset for drift monitoring.

### Required Changes (example using loan model)

``` python
features = ['Income', 'LoanAmount', 'CreditHistory']

# Export baseline dataset for monitoring
X[features].to_csv('data/baseline_data.csv', index=False)

# Save trained model
joblib.dump(model, 'models/loan_model.joblib')
```

------------------------------------------------------------------------

## B. app.py (Serving Layer)

### Goal

Ensure UI inputs match model feature schema.

### Initialize Monitoring SDK

``` python
monitor = MLExporter(port=8000, baseline_path='data/baseline_data.csv')
```

### Map UI Inputs to DataFrame (example using loan model)

``` python
input_data = [
    float(request.form['f1']),
    float(request.form['f2']),
    float(request.form['f3'])
]

df = pd.DataFrame([input_data],
                  columns=['Income', 'LoanAmount', 'CreditHistory'])
```

------------------------------------------------------------------------

## C. exporter.py (Monitoring Engine)

### Goal

Update drift logic and monitoring labels.

### Update Drift Index (LoanAmount → Index 1) (example using loan model)

``` python
def check_drift_and_features(self, live_data):
    stat, _ = ks_2samp(self.baseline_df.iloc[:, 1],
                       live_data.iloc[:, 1])
    DRIFT_SCORE.set(stat)

    FEATURE_MIN.labels(feature_name='loan_amount')                .set(live_data.iloc[0, 1])
```

------------------------------------------------------------------------

# 4. Setup & Deployment (Step-by-Step)

## Step 1: Generate Model & Baseline

Run locally:

    python model_setup.py

This creates:

-   `models/loan_model.joblib`
-   `data/baseline_data.csv`

------------------------------------------------------------------------

## Step 2: Build & Start Services

    docker-compose up --build

This launches:

-   Flask App
-   Prometheus
-   Grafana
-   Monitoring SDK

------------------------------------------------------------------------

## Step 3: Access Services

| Service               | URL                           | Purpose                                   |
|-----------------------|-------------------------------|-------------------------------------------|
| Loan Predictor UI     | http://localhost:5000        | Submit predictions and provide feedback   |
| Metrics Stream        | http://localhost:8000/metrics| View raw Prometheus metrics               |
| Prometheus UI         | http://localhost:9090        | Query and analyze metrics manually        |
| Grafana Dashboard     | http://localhost:3000        | Visual monitoring and dashboard insights  |

---

# 5. Metrics & Visualization Strategy

The monitoring system groups metrics into four key quadrants:

| Category      | Dashboard Panels                    | Mathematical / Technical Basis |
|--------------|-------------------------------------|---------------------------------|
| ML Quality   | F1-Score, Precision, Recall         | F1 = 2 × (Precision × Recall) / (Precision + Recall) |
| Data Drift   | KS-Score                            | Dₙ,ₘ = sup\|F₁(x) − F₂(x)\| (Kolmogorov–Smirnov statistic) |
| Operational  | Latency Histogram, Error Count      | HTTP request duration tracking and error rate calculation |
| Resource     | CPU, RAM, Disk Usage                | Real-time system metrics via `psutil` |


------------------------------------------------------------------------

# Professional Modification Guidelines

##  Adding New Features (Example: Age)

1.  Add column in `model_setup.py`
2.  Regenerate baseline CSV
3.  Update DataFrame columns in `app.py`
4.  Adjust drift index in `exporter.py` if needed

------------------------------------------------------------------------

## Threshold Management (Grafana)

Inside `dashboard.json`:

-   Accuracy & F1 → Green if \> 0.8
-   Drift Score → Red if high
-   Latency → Red if above SLA

------------------------------------------------------------------------

## Simulation Testing

### Trigger Drift Alert

- Input an extremely high LoanAmount (e.g., 900000)

### Trigger Accuracy Drop

- Provide incorrect feedback label intentionally.

------------------------------------------------------------------------

# Final Notes

This framework enables:

-   Internal ML Observability
-   Model Performance Auditing
-   Drift Monitoring
-   Production-Grade Monitoring Integration

It is modular and scalable for future ML models.
