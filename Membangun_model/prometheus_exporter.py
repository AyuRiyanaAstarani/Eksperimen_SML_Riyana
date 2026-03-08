from fastapi import FastAPI, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
import psutil
import numpy as np
import os
import mlflow.pyfunc

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model_path = os.path.join(
    BASE_DIR,
    "mlruns",
    "2",
    "models",
    "model"
)

# Metrics

# total request
request_count = Counter(
    "http_requests_total",
    "Total jumlah request model"
)


prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Waktu prediksi model"
)


cpu_usage = Gauge(
    "system_cpu_usage",
    "CPU usage percentage"
)


memory_usage = Gauge(
    "system_memory_usage",
    "Memory usage percentage"
)


disk_usage = Gauge(
    "system_disk_usage",
    "Disk usage percentage"
)


@app.get("/predict")
def predict():

    request_count.inc()

    start = time.time()

    time.sleep(0.1)

    latency = time.time() - start
    prediction_latency.observe(latency)

    cpu_usage.set(psutil.cpu_percent())
    memory_usage.set(psutil.virtual_memory().percent)
    disk_usage.set(psutil.disk_usage('/').percent)

    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)