from prometheus_client import start_http_server, Counter
import time

prediction_counter = Counter('model_predictions', 'Total Predictions')

def predict():
    prediction_counter.inc()

if __name__ == "__main__":
    start_http_server(8000)
    while True:
        predict()
        time.sleep(5)