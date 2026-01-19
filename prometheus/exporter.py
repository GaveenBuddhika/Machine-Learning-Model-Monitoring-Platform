import psutil
from prometheus_client import start_http_server, Gauge, Histogram
from scipy.stats import ks_2samp
import pandas as pd

# External Metrics (Operational Health)
LATENCY = Histogram('model_latency_seconds', 'Prediction latency')
CPU_USAGE = Gauge('model_cpu_usage', 'CPU usage percentage')
MEM_USAGE = Gauge('model_memory_usage', 'Memory usage bytes')

# Internal Metrics (Model Performance)
DRIFT_GAUGE = Gauge('model_drift_score', 'K-S Test drift score (0-1)')
PREDICTION_VAL = Gauge('model_prediction_value', 'Last prediction result')

class MLExporter:
    def __init__(self, port=8000, baseline_path='data/baseline_data.csv'):
        self.baseline_df = pd.read_csv(baseline_path)
        start_http_server(port)
        print(f"Metrics Exporter running on port {port}")

    def track_health(self):
        CPU_USAGE.set(psutil.cpu_percent())
        MEM_USAGE.set(psutil.Process().memory_info().rss)

    def check_drift(self, live_data):
        # Using K-S Test algorithm for drift detection
        stat, p = ks_2samp(self.baseline_df.iloc[:, 0], live_data.iloc[:, 0])
        DRIFT_GAUGE.set(stat)
        return stat