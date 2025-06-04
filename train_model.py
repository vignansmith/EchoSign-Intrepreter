import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Dataset path
DATASET_PATH = "asl_dataset/"
IMG_SIZE = 64  # Resize images to 64x64
BATCH_SIZE = 32
EPOCHS = 20

# Load dataset
def load_dataset():
    images = []
    labels = []
    gesture_names = os.listdir(DATASET_PATH)
    
    for gesture in gesture_names:
        gesture_path = os.path.join(DATASET_PATH, gesture)
        if not os.path.isdir(gesture_path):
            continue
        
        for img_name in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize
            images.append(img)
            labels.append(gesture)
    
    images = np.array(images) / 255.0  # Normalize pixel values
    images = np.expand_dims(images, axis=-1)  # Add channel dimension
    
    # Convert labels to numerical values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)  # One-hot encoding
    
    return images, labels, label_encoder

# Build CNN model
def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Load dataset
print("Loading dataset...")
X, y, label_encoder = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train model
num_classes = y.shape[1]
model = build_model(num_classes)
model.summary()

print("Training model...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

# Save the model and label encoder
model.save("asl_model.h5")
np.save("labels.npy", label_encoder.classes_)
print("Model and labels saved successfully!")
