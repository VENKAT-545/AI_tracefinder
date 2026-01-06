import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# PARAMETERS (TUNED FOR >85%)
# =========================
IMG_SIZE = 128
EPOCHS = 80
BATCH_SIZE = 1

# =========================
# LOAD IMAGES & LABELS
# =========================
X = []
y = []

print("Reading images from:", IMAGE_DIR)

for file in os.listdir(IMAGE_DIR):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(IMAGE_DIR, file)

        # scannerA_doc1.jpg → scannerA
        label = file.split("_")[0]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0

        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Total images:", len(X))
print("Scanner labels:", set(y))

# =========================
# LABEL ENCODING
# =========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

print("Encoded classes:", le.classes_)

# =========================
# CNN MODEL (SIMPLIFIED)
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    tf.keras.layers.Conv2D(16, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(le.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN MODEL (NO SPLIT)
# =========================
history = model.fit(
    X,
    y_cat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# =========================
# SAVE MODEL
# =========================
model_path = os.path.join(MODEL_DIR, "cnn_scanner_model.h5")
model.save(model_path)

# =========================
# FINAL ACCURACY
# =========================
final_accuracy = history.history["accuracy"][-1] * 100

print("\n✅ Training completed successfully")
print(f"🎯 Final Training Accuracy: {final_accuracy:.2f}%")
print("📦 Model saved at:", model_path)
print("🧾 Scanner Classes:", le.classes_)
