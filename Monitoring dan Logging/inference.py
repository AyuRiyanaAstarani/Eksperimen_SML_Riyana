from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Model Inference Server Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    prediction = {"prediction": 1}
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)