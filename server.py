from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import io
import time

app = Flask(__name__)

# โหลดโมเดล
model = tf.keras.models.load_model("converted_keras/keras_model.h5")

# Telegram
BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

start_time = None

def send_telegram(msg):
    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": msg}
    )

@app.route("/")
def home():
    return "AI Server Running"

@app.route("/predict", methods=["POST"])
def predict():
    global start_time

    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files["image"]

    # เตรียมภาพ (ปรับขนาดให้ตรงกับโมเดลคุณ)
    img = Image.open(file.stream).convert("RGB").resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    prob = float(prediction[0][1])  # class 1 = eyes_closing

    now = time.time()

    if prob > 0.6:
        if start_time is None:
            start_time = now

        if now - start_time > 10:
            send_telegram("⚠️ eyes_closing 10s")
            start_time = None
    else:
        start_time = None

    return jsonify({"probability": prob})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
