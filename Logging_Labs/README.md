# Lab 2: Advanced Logging for MLOps  
Student : Gayatri Nair
Date : November 2025  

# 1. Overview  
This lab implements a production-style logging system for a Machine Learning pipeline using Python.  
Compared to earlier basic logging versions, this upgrade introduces a complete MLOps-grade setup with:

- Multiple specialized loggers assigned to specific log files  
- JSON-based audit logging (`app_json.log`)  
- NumPy-based synthetic data for training  
- Data drift detection (`data_drift.log`)  
- Prediction logging (`predictions.log`)  
- Model monitoring metrics (`monitoring.log`)  
- API request logging (`api_requests.log`)  
- Exception logging with full stack traces (`errors.log`)  

# 2. Project Structure

```
Logging_Labs/
│
├── logging_demo.py        # Main script containing all loggers
├── requirements.txt       # Required Python packages
│
├── ml_pipeline.log        # Training & validation logs (generated)
├── predictions.log        # Model prediction logs (generated)
├── data_drift.log         # Drift detection logs (generated)
├── monitoring.log         # Model service metrics (generated)
├── api_requests.log       # API call logs (generated)
├── app_json.log           # JSON audit logs (generated)
└── errors.log             # Exception stack trace logs (generated)
```

All logs are generated separately, just like in real ML deployment environments.

# 3. What This Lab Demonstrates  

Multi-logger Architecture  
Each part of the ML workflow writes to a dedicated log file:

- Training events : ml_pipeline.log  
- Prediction events : predictions.log  
- Data drift checks : data_drift.log  
- Model monitoring metrics : monitoring.log  
- API calls : api_requests.log  
- Audit events in JSON format : app_json.log  
- Errors and exceptions : errors.log

# 4. Improvements Over Previous Versions  

a. NumPy-based Synthetic Data  
The training pipeline generates Gaussian synthetic data and logs metrics such as mean and standard deviation into ml_pipeline.log

b. Data Drift Detection  
Simulated batch shifts are logged as drift or no-drift messages inside data_drift.log

c. Prediction Logging  
Every prediction logs:  
- input features  
- decision (approve/deny)  
- confidence score  
These go into predictions.log

d. JSON Audit Logging  
Audit events such as `ml_pipeline_started` and `ml_pipeline_completed` are logged into app_json.log using a custom JSON formatter.

e. Model Monitoring  
Simulated service metrics (requests, errors, average latency) are stored in monitoring.log.

f. Exception Handling  
Runtime errors like division by zero and missing files are written with full tracebacks to errors.log.

# 5. Log Files Generated  

The script automatically generates the following log files:

- **ml_pipeline.log** : training loop, epochs, validation accuracy  
- **predictions.log** : inference logs  
- **data_drift.log** : drift monitoring messages  
- **monitoring.log** : service-level metrics  
- **api_requests.log** : API request statuses  
- **app_json.log** : JSON audit events  
- **errors.log** : full exception stack traces  

# 6. How to Run (Windows)

```powershell
cd Logging_Labs
python logging_demo.py
