import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_scanner_model.h5")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

IMG_SIZE = 128

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# LOAD IMAGES & LABELS
# =========================
X = []
y = []

for file in os.listdir(IMAGE_DIR):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(IMAGE_DIR, file)
        label = file.split("_")[0]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0

        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# =========================
# PREDICTIONS
# =========================
preds = model.predict(X)
y_pred = np.argmax(preds, axis=1)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_encoded, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Scanner Identification")

cm_path = os.path.join(RESULT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.show()

# =========================
# REPORT
# =========================
print("\n📊 Classification Report")
print(classification_report(y_encoded, y_pred, target_names=le.classes_))
print("✅ Confusion matrix saved at:", cm_path)
