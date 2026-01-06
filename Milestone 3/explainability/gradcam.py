import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

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
# REBUILD MODEL (FUNCTIONAL)
# =========================
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = tf.keras.layers.Conv2D(16, (3,3), activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D(2,2)(x)

x = tf.keras.layers.Conv2D(32, (3,3), activation="relu", name="last_conv")(x)
x = tf.keras.layers.MaxPooling2D(2,2)(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

functional_model = tf.keras.Model(inputs, outputs)

# =========================
# LOAD WEIGHTS
# =========================
functional_model.load_weights(MODEL_PATH)

# =========================
# GRAD MODEL
# =========================
grad_model = tf.keras.Model(
    inputs=functional_model.inputs,
    outputs=[
        functional_model.get_layer("last_conv").output,
        functional_model.output
    ]
)

# =========================
# LOAD IMAGE
# =========================
image_file = os.listdir(IMAGE_DIR)[0]
img_path = os.path.join(IMAGE_DIR, image_file)

img = cv2.imread(img_path)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_norm = img.astype("float32") / 255.0
input_tensor = tf.expand_dims(img_norm, axis=0)

# =========================
# COMPUTE GRAD-CAM
# =========================
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(input_tensor)
    class_index = tf.argmax(predictions[0])
    loss = predictions[:, class_index]

grads = tape.gradient(loss, conv_outputs)

# Global average pooling
pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

conv_outputs = conv_outputs[0]
heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

heatmap = tf.maximum(heatmap, 0)
heatmap /= tf.reduce_max(heatmap)

# =========================
# VISUALIZE
# =========================
heatmap = heatmap.numpy()
heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

output_path = os.path.join(RESULT_DIR, "gradcam_output.png")
cv2.imwrite(output_path, overlay)

plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM – Scanner Artifact Visualization")
plt.axis("off")
plt.show()

print("✅ Grad-CAM generated successfully")
print("📍 Saved at:", output_path)
