# Sistem Monitoring Model Machine Learning

Project ini merupakan implementasi pipeline Machine Learning yang dilengkapi dengan monitoring dan alerting.

## Dataset
Dataset yang digunakan adalah **Churn_Modelling.csv** yang berisi data pelanggan bank.

## Tahapan Project

### 1. Data Preprocessing
Data diproses menggunakan:
- automate_Riyana.py
- Eksperimen_Riyana.ipynb

Output:
- train_preprocessed.csv
- test_preprocessed.csv

### 2. Model Training
Model dilatih menggunakan:
- modelling.py

Hyperparameter tuning:
- modelling_tuning.py

### 3. Model Serving
Model diserve menggunakan FastAPI melalui:
- inference.py

Endpoint: http://localhost:8000/predict


### 4. Monitoring
Monitoring model menggunakan:
- Prometheus
- Grafana

Metric yang dimonitor:
- http_requests_total
- prediction_latency_seconds
- system_cpu_usage

### 5. Alerting
Alert dibuat menggunakan Grafana jika jumlah request model melebihi threshold tertentu.

## Tools yang digunakan

- Python
- FastAPI
- MLflow
- Prometheus
- Grafana
