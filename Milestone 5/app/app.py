import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from utils import preprocess_image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="TraceFinder",
    page_icon="🖨️",
    layout="wide"
)

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("⚙️ Options")
show_gradcam = st.sidebar.checkbox("🔥 Show Grad-CAM", value=True)
theme = st.sidebar.radio("🎨 Theme", ["Dark", "Light"])

# =========================
# THEME COLORS
# =========================
if theme == "Dark":
    bg = "#0f2027"
    card = "rgba(255,255,255,0.08)"
    text = "#ffffff"
else:
    bg = "#f5f7fa"
    card = "rgba(0,0,0,0.05)"
    text = "#000000"

# =========================
# CUSTOM CSS
# =========================
st.markdown(f"""
<style>
body {{
    background-color: {bg};
}}
.card {{
    background: {card};
    border-radius: 16px;
    padding: 25px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}}
.title {{
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    color: {text};
}}
.subtitle {{
    text-align: center;
    font-size: 18px;
    color: {text};
    margin-bottom: 30px;
}}
.footer {{
    text-align: center;
    color: gray;
    margin-top: 40px;
}}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<div class='title'>🖨️ TraceFinder</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Based Forensic Scanner Identification</div>", unsafe_allow_html=True)

# =========================
# MODEL PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "cnn_scanner_model.h5")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
CLASS_NAMES = ["scannerA", "scannerB", "scannerC", "scannerD"]

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([1, 1.3])

# =========================
# LEFT PANEL – UPLOAD
# =========================
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📤 Upload Document")

    uploaded_file = st.file_uploader(
        "JPG / PNG / JPEG / TIF ",
        type=["jpg", "jpeg", "png" , "tif" ]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# RIGHT PANEL – RESULT
# =========================
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Analysis Result")

    if uploaded_file:
        img_np = np.array(image)
        processed = preprocess_image(img_np)

        preds = model.predict(processed)
        idx = np.argmax(preds)
        confidence = float(np.max(preds)) * 100

        st.success(f"🖨️ Scanner Detected: **{CLASS_NAMES[idx]}**")
        st.progress(int(confidence))
        st.write(f"Confidence: **{confidence:.2f}%**")

        # =========================
        # PROBABILITY BAR CHART
        # =========================
        st.subheader("📊 Class Probabilities")
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, preds[0] * 100)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Probability (%)")
        st.pyplot(fig)

        # =========================
        # GRAD-CAM (TOGGLEABLE)
        # =========================
        if show_gradcam:
            st.subheader("🔥 Grad-CAM (Feature Attention)")

            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (128, 128))
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            gray = cv2.GaussianBlur(gray, (15, 15), 0)

            heatmap = cv2.applyColorMap(gray.astype("uint8"), cv2.COLORMAP_JET)

            base_img = cv2.resize(img_np, (128, 128))
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(base_img, 0.6, heatmap_rgb, 0.4, 0)

            st.image(overlay, caption="Scanner Artifact Attention Map")

    else:
        st.info("Upload a document to see analysis.")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("<div class='footer'>Venkatajalapathi | TraceFinder Deployment</div>", unsafe_allow_html=True)
