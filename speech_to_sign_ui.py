import sys
import os
import cv2
import numpy as np
import speech_recognition as sr
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap
from PyQt5.QtCore import Qt, QRunnable, QThreadPool, pyqtSignal, QObject

# Load trained model and labels
MODEL_PATH = "asl_model.h5"
LABELS_PATH = "labels.npy"
SIGN_LANGUAGE_PATH = "asl_dataset/"  # Folder containing gesture images

model = tf.keras.models.load_model(MODEL_PATH)
label_classes = np.load(LABELS_PATH)

class SpeechWorkerSignals(QObject):
    result = pyqtSignal(str)
    error = pyqtSignal(str)

class SpeechWorker(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = SpeechWorkerSignals()

    def run(self):
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                print("Listening...")
                self.signals.result.emit("🎙 Listening...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
            
            text = recognizer.recognize_google(audio).lower()
            print(f"Recognized: {text}")
            self.signals.result.emit(text)
        except sr.UnknownValueError:
            print("Could not understand audio.")
            self.signals.error.emit("⚠️ Could not understand the audio.")
        except sr.RequestError as e:
            print(f"Request failed: {e}")
            self.signals.error.emit("❌ Could not request results, check your internet connection.")

class SpeechToSignApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.threadpool = QThreadPool()
        self.worker = None

    def initUI(self):
        self.setWindowTitle("EchoSign Interpreter")
        self.setGeometry(300, 300, 1000, 500)
        
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#282c36"))
        self.setPalette(palette)
        
        self.label = QLabel("Click the button and speak", self)
        self.label.setFont(QFont("Arial", 14))
        self.label.setStyleSheet("color: white; text-align: center;")
        self.label.setAlignment(Qt.AlignCenter)
        
        self.sign_layout = QHBoxLayout()
        
        self.start_button = QPushButton("🎤 Start Speaking", self)
        self.style_button(self.start_button, "#61afef")
        self.start_button.clicked.connect(self.start_speech_recognition)
        
        self.stop_button = QPushButton("⛔ Stop", self)
        self.style_button(self.stop_button, "#e06c75")
        self.stop_button.clicked.connect(self.stop_speech_recognition)
        
        self.exit_button = QPushButton("❌ Exit", self)
        self.style_button(self.exit_button, "#98c379")
        self.exit_button.clicked.connect(self.close)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.exit_button)
        
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(self.sign_layout)
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def style_button(self, button, color):
        button.setFont(QFont("Arial", 12, QFont.Bold))
        button.setStyleSheet(f"background-color: {color}; color: white; padding: 10px; border-radius: 10px;")
        button.setCursor(Qt.PointingHandCursor)
    
    def start_speech_recognition(self):
        if self.worker is None:
            self.worker = SpeechWorker()
            self.worker.signals.result.connect(self.update_sign_language)
            self.worker.signals.error.connect(self.update_label)
            self.threadpool.start(self.worker)
    
    def stop_speech_recognition(self):
        if self.worker:
            self.label.setText("⛔ Speech recognition stopped.")
            self.worker = None
    
    def update_label(self, text):
        self.label.setText(text)
    
    def update_sign_language(self, text):
        self.label.setText(f"🗣 You said: {text}")
        words = text.split()
        
        # Clear previous images
        while self.sign_layout.count():
            item = self.sign_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        for word in words:
            word = word.lower()
            image_path = self.find_gesture_image(word)
            if image_path:
                pixmap = QPixmap(image_path)
                img_label = QLabel()
                img_label.setPixmap(pixmap)
                self.sign_layout.addWidget(img_label)
    
    def find_gesture_image(self, word):
        """Finds the closest matching gesture image for a word."""
        for gesture in label_classes:
            if word == gesture.lower():
                image_path = os.path.join(SIGN_LANGUAGE_PATH, gesture, "0.jpg")  # Assuming first image represents the gesture
                if os.path.exists(image_path):
                    return image_path
        return None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpeechToSignApp()
    window.show()
    sys.exit(app.exec_())
