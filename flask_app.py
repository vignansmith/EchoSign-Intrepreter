from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import speech_recognition as sr
import os

app = Flask(__name__)

# Load trained ASL model
MODEL_PATH = "asl_model.h5"
LABELS_PATH = "labels.npy"
IMG_SIZE = 64  # Image size used for training

model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)

# Speech-to-text function
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text.lower()
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Speech recognition service error"

# Route to process speech input
@app.route("/speech-to-sign", methods=["POST"])
def speech_to_sign():
    text = recognize_speech()
    response = {"text": text, "signs": text.split()}  # Simple word splitting for now
    return jsonify(response)

# Route to process hand gesture image
@app.route("/predict-gesture", methods=["POST"])
def predict_gesture():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Reshape for model
    
    prediction = model.predict(img)
    predicted_label = labels[np.argmax(prediction)]
    return jsonify({"gesture": predicted_label})

if __name__ == "__main__":
    app.run(debug=True)
