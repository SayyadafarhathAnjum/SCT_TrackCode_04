import streamlit as st
from PIL import Image
import numpy as np
import io
from gesture_model import process_image, GESTURE_LABELS

st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="🤚",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #f0f0f0; }
    .title-box {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 28px;
        text-align: center;
    }
    .title-box h1 { font-family: 'Space Mono', monospace; font-size: 2rem; color: #a78bfa; margin: 0; }
    .title-box p { color: #c4b5fd; margin: 8px 0 0; font-size: 0.95rem; }
    .result-card {
        background: rgba(167,139,250,0.12);
        border: 1px solid #a78bfa;
        border-radius: 12px;
        padding: 20px 28px;
        margin-top: 20px;
        text-align: center;
    }
    .gesture-label { font-size: 2rem; font-weight: 700; color: #c4b5fd; }
    .confidence-label { font-size: 0.9rem; color: #a0aec0; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-box">
    <h1>🤚 Hand Gesture Recognition</h1>
    <p>Upload a hand image — MediaPipe detects and classifies the gesture</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload a hand image (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    image_bytes = uploaded.read()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Image**")
        original = Image.open(io.BytesIO(image_bytes))
        st.image(original, use_container_width=True)
    with st.spinner("Detecting gesture..."):
        annotated_np, gesture_label, confidence = process_image(image_bytes)
    with col2:
        st.markdown("**Landmark Detection**")
        if annotated_np is not None:
            st.image(annotated_np, use_container_width=True)
        else:
            st.warning("Could not process image.")
    st.markdown(f"""
    <div class="result-card">
        <div class="gesture-label">{gesture_label}</div>
        <div class="confidence-label">Confidence: {confidence}</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("👆 Upload a clear image of a hand to get started.")

st.markdown("---")
st.markdown("<center style='color:#718096;font-size:0.8rem;'>SkillCraft Technology · Task 04</center>", unsafe_allow_html=True)