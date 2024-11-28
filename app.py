from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import requests
import os
from google.cloud import firestore
from datetime import datetime
import pytz

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "fabled-rookery-438514-v3-c25bd3e6b38b.json"

# Inisialisasi Flask
app = Flask(__name__)

# URL publik untuk model TFLite dan scaler
MODEL_URL = "https://storage.googleapis.com/bumil/modelml.tflite"
SCALER_URL = "https://storage.googleapis.com/bumil/scaler.pkl"

# Lokasi sementara untuk menyimpan file yang diunduh
MODEL_PATH = "modelml.tflite"
SCALER_PATH = "scaler.pkl"

# Fungsi untuk mengunduh file dari Cloud Storage
def download_file(url, destination):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"File berhasil diunduh: {destination}")
    else:
        raise Exception(f"Gagal mengunduh file dari {url}. Status code: {response.status_code}")

# Unduh model dan scaler jika belum ada
if not os.path.exists(MODEL_PATH):
    download_file(MODEL_URL, MODEL_PATH)

if not os.path.exists(SCALER_PATH):
    download_file(SCALER_URL, SCALER_PATH)

# Muat model TFLite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Muat scaler yang telah disimpan
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Mendapatkan input dan output details untuk interpreter
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Inisialisasi Firestore
db = firestore.Client()

@app.route("/", methods=["GET"])
def home():
    return "TensorFlow Lite Web Service is Running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Pastikan request memiliki data JSON
        data = request.get_json()

        # Ambil input data dari request
        input_data = np.array(data["input"]).reshape(1, -1)
        # np.array(data["input"]).flatten().tolist()
        # Normalisasi menggunakan scaler
        input_data = scaler.transform(input_data)

        # Memberikan input ke model TFLite
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))

        # Melakukan inferensi
        interpreter.invoke()

        # Ambil hasil output dari model
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Prediksi dan kategori
        prediction = output_data.tolist()
        risk_category = ['Low Risk', 'Mid Risk', 'High Risk'][np.argmax(output_data)]

        flat_input = np.array(data["input"]).flatten().tolist()
        flat_prediction = np.array(prediction).flatten().tolist()

        wib_timezone = pytz.timezone("Asia/Jakarta")
        wib_time = datetime.now(wib_timezone).isoformat()

        # Simpan hasil prediksi ke Firestore
        prediction_data = {
            "input": flat_input,
            "prediction": flat_prediction,
            "risk_category": risk_category,
            "timestamp": wib_time
        }

        db.collection("predictions").add(prediction_data)

        return jsonify({
            "prediction": flat_prediction,
            "risk_category": risk_category
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
