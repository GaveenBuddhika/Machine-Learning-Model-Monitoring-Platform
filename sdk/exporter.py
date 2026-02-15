import psutil
import pandas as pd
from prometheus_client import start_http_server, Gauge, Histogram, Counter
from scipy.stats import ks_2samp

# --- [1] Infrastructure Metrics ---
CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
MEM_USAGE = Gauge('system_memory_usage_bytes', 'Memory usage in bytes')
DISK_USAGE = Gauge('system_disk_usage_percent', 'Disk storage usage percentage')

# --- [2] Operational Metrics ---
LATENCY = Histogram('model_latency_seconds', 'Model internal processing time')
RESPONSE_TIME = Histogram('api_response_time_seconds', 'Total end-to-end response time')
PREDICTION_TOTAL = Counter('model_prediction_total', 'Total number of successful predictions')
ERROR_COUNT = Counter('model_prediction_error_total', 'Total number of prediction errors')

# --- [3] ML Quality Metrics ---
DRIFT_SCORE = Gauge('model_drift_score', 'Data drift score using KS Test')
ACCURACY_SCORE = Gauge('model_accuracy_score', 'Real-time accuracy of the model')
PRECISION_SCORE = Gauge('model_precision_score', 'Real-time model precision')
RECALL_SCORE = Gauge('model_recall_score', 'Real-time model recall')
F1_SCORE = Gauge('model_f1_score', 'Real-time model F1-Score')

# --- [4] Data Feature Metrics (Updated Label to 'loan_amount') ---
FEATURE_MIN = Gauge('feature_min_value', 'Minimum value of the current feature batch', ['feature_name'])
FEATURE_MAX = Gauge('feature_max_value', 'Maximum value of the current feature batch', ['feature_name'])
PRED_PROBABILITY = Histogram('model_prediction_probability', 'Confidence score of the prediction')

class MLExporter:
    def __init__(self, port=8000, baseline_path='data/loan_baseline.csv'):
        # Load baseline for drift detection (Now using loan_baseline.csv)
        self.baseline_df = pd.read_csv(baseline_path)
        
        # Initialize Confusion Matrix counters
        self.tp = 0; self.fp = 0; self.tn = 0; self.fn = 0
        
        start_http_server(port)
        print(f"Loan Monitoring SDK (Stage 1) running on port {port}")

    def track_system_health(self):
        """Captures hardware performance metrics."""
        CPU_USAGE.set(psutil.cpu_percent())
        MEM_USAGE.set(psutil.virtual_memory().used)
        DISK_USAGE.set(psutil.disk_usage('/').percent)

    def log_error(self):
        """Increments error counter."""
        ERROR_COUNT.inc()

    def check_drift_and_features(self, live_data):
        """
        Calculates drift and tracks feature ranges for LoanAmount (Index 1).
        In your Loan Setup: Index 0=Income, 1=LoanAmount, 2=CreditHistory
        """
        # Compare live LoanAmount distribution vs baseline LoanAmount distribution
        stat, _ = ks_2samp(self.baseline_df.iloc[:, 1], live_data.iloc[:, 1])
        DRIFT_SCORE.set(stat)
        
        # Track Min/Max for LoanAmount
        val = live_data.iloc[0, 1] 
        FEATURE_MIN.labels(feature_name='loan_amount').set(val)
        FEATURE_MAX.labels(feature_name='loan_amount').set(val)
        
        PREDICTION_TOTAL.inc()

    def track_performance_metrics(self, prediction, actual):
        """Updates Confusion Matrix and exports all ML math metrics."""
        # 1. Update Confusion Matrix (1: Approved, 0: Rejected)
        if prediction == 1 and actual == 1: self.tp += 1
        elif prediction == 1 and actual == 0: self.fp += 1
        elif prediction == 0 and actual == 0: self.tn += 1
        elif prediction == 0 and actual == 1: self.fn += 1

        # 2. Update Basic Accuracy
        ACCURACY_SCORE.set(1 if prediction == actual else 0)

        # 3. Calculate Precision & Recall
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        
        # 4. Calculate F1-Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # 5. Export to Prometheus
        PRECISION_SCORE.set(precision)
        RECALL_SCORE.set(recall)
        F1_SCORE.set(f1)