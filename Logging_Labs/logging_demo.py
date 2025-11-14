"""
Lab 2: Logging for MLOps
Student: Gayatri
Date: November 2025

Explanation of Improvements:
This final version of my logging script is completely different and much more advanced than the previous two versions. 
Earlier scripts only demonstrated simple logging such as training messages, API logs, JSON logs, and basic exceptions. 
This new version implements a full MLOps-style logging architecture with multiple specialized loggers and NumPy integration. 
It generates synthetic training data using NumPy and logs real statistical metrics (mean and std), adds a dedicated 
Prediction Logger, implements Data Drift Detection using real NumPy distributions, and introduces a Model Monitoring Logger 
to track service-level metrics. It also produces structured JSON audit logs and separate error logs. The final script creates 
multiple log files (ml_pipeline.log, predictions.log, data_drift.log, monitoring.log, api_requests.log, app_json.log, errors.log)
making it much more realistic and production-focused than earlier versions.
"""

import logging
import logging.handlers
import numpy as np     # <-- NEW EXTERNAL LIBRARY
import json
from datetime import datetime
import time
import random


# ===========================
# 1. JSON FORMATTER
# ===========================

class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        return json.dumps(data)


# ===========================
# 2. INIT LOGGING
# ===========================

def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console logger for root
    if not root.handlers:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            "%Y-%m-%d %H:%M:%S"
        ))
        root.addHandler(console)

    # File loggers
    file_configs = [
        ("ml.train", "ml_pipeline.log"),
        ("api", "api_requests.log"),
        ("ml.prediction", "predictions.log"),
        ("ml.drift", "data_drift.log"),
        ("ml.monitor", "monitoring.log"),
        ("json.audit", "app_json.log")
    ]

    for logger_name, file_name in file_configs:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            h = logging.FileHandler(file_name)
            if logger_name == "json.audit":
                h.setFormatter(JsonFormatter())
            else:
                h.setFormatter(logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s"
                ))
            logger.addHandler(h)

    # Exception logger (console + file)
    exc = logging.getLogger("errors")
    exc.setLevel(logging.ERROR)
    if not exc.handlers:
        # Console output for errors
        eh_console = logging.StreamHandler()
        eh_console.setFormatter(logging.Formatter(
            "%(levelname)s - %(message)s"
        ))
        exc.addHandler(eh_console)

        # File output for errors
        eh_file = logging.FileHandler("errors.log")
        eh_file.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        exc.addHandler(eh_file)


# ===========================
# 3. TRAINING + VALIDATION (NumPy Added)
# ===========================

def run_training():
    log = logging.getLogger("ml.train")

    log.info("Training started using NumPy synthetic data")

    # Generate synthetic data
    data = np.random.normal(50, 10, size=10000)
    mean, std = np.mean(data), np.std(data)

    log.info(f"Training data statistics: mean={mean:.2f}, std={std:.2f}")

    # Fake training loop
    for epoch in range(1, 4):
        time.sleep(0.1)
        loss = 0.5 / epoch
        acc = 0.7 + 0.1 * epoch
        log.info(f"Epoch {epoch}/3 : loss={loss:.4f}, acc={acc:.4f}")

    log.info("Validation accuracy: 0.91")
    log.info("Model saved to ./models/model_v4.pkl")

    return mean


# ===========================
# 4. DATA DRIFT DETECTION (NumPy Added)
# ===========================

def simulate_drift_monitoring(training_mean):
    drift_logger = logging.getLogger("ml.drift")

    for i in range(4):
        drift_amt = random.uniform(-10, 15)
        new_batch = np.random.normal(50 + drift_amt, 10, size=500)

        batch_mean = np.mean(new_batch)
        drift_score = abs(batch_mean - training_mean)

        if drift_score > 8:
            drift_logger.warning(
                f"Drift detected! batch_mean={batch_mean:.2f}, drift_score={drift_score:.2f}"
            )
        else:
            drift_logger.info(
                f"No drift. batch_mean={batch_mean:.2f}, drift_score={drift_score:.2f}"
            )


# ===========================
# 5. PREDICTION LOGGER (NumPy Added)
# ===========================

def log_prediction(age, income):
    pred = logging.getLogger("ml.prediction")

    features = np.array([age, income], dtype=float)
    score = (features[0] / 100) + (features[1] / 100000)

    decision = "approve" if score > 0.4 else "deny"

    pred.info(f"Input features: {features.tolist()}")
    pred.info(f"Model output: decision={decision}, score={score:.3f}")


# ===========================
# 6. MODEL MONITORING
# ===========================

def log_model_metrics():
    m = logging.getLogger("ml.monitor")
    m.info("Service metrics: requests=150, errors=2, avg_latency=48ms")


# ===========================
# 7. API REQUESTS
# ===========================

def log_api_call(endpoint, status):
    api = logging.getLogger("api")
    api.info(f"POST {endpoint} - Status={status}")


# ===========================
# 8. JSON AUDIT EVENTS
# ===========================

def log_audit_event(action, user="system"):
    audit = logging.getLogger("json.audit")
    audit.info(f"Audit event: user={user}, action={action}")


# ===========================
# 9. EXCEPTIONS
# ===========================

def demo_exceptions():
    exc = logging.getLogger("errors")
    try:
        1 / 0
    except Exception:
        exc.exception("ZeroDivisionError occurred")

    try:
        open("missing_file.txt")
    except Exception:
        exc.exception("Missing file error")


# ===========================
# MAIN
# ===========================

if __name__ == "__main__":
    init_logging()

    # JSON audit logs around the pipeline
    log_audit_event("ml_pipeline_started", user="service-account")

    mean = run_training()
    simulate_drift_monitoring(mean)

    log_prediction(30, 55000)
    log_prediction(62, 24000)

    log_api_call("/predict", 200)
    log_api_call("/predict", 404)

    log_model_metrics()
    demo_exceptions()

    log_audit_event("ml_pipeline_completed", user="service-account")

    print("Generated logs:")
    print("- ml_pipeline.log")
    print("- data_drift.log")
    print("- predictions.log")
    print("- api_requests.log")
    print("- monitoring.log")
    print("- app_json.log")
    print("- errors.log")
