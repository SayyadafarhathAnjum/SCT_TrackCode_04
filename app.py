import streamlit as st
from PIL import Image
import numpy as np
import io
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

GESTURE_LABELS = {
    "thumbs_up": "👍 Thumbs Up",
    "thumbs_down": "👎 Thumbs Down",
    "open_hand": "✋ Open Hand",
    "fist": "✊ Fist",
    "peace": "✌️ Peace",
    "pointing": "☝️ Pointing",
    "ok": "👌 OK",
    "unknown": "❓ Unknown"
}

def get_finger_states(landmarks):
    fingers = []
    fingers.append(landmarks[4].x < landmarks[3].x)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip_idx, pip_idx in zip(tips, pips):
        fingers.append(landmarks[tip_idx].y < landmarks[pip_idx].y)
    return fingers

def classify_gesture(landmarks):
    f = get_finger_states(landmarks)
    thumb, index, middle, ring, pinky = f
    if all(f):
        return "open_hand"
    if not any(f):
        return "fist"
    if thumb and not index and not middle and not ring and not pinky:
        return "thumbs_up"
    if not thumb and index and not middle and not ring and not pinky:
        return "pointing"
    if not thumb and index and middle and not ring and not pinky:
        return "peace"
    if thumb and index and not middle and not ring and not pinky:
        return "ok"
    if landmarks[4].y > landmarks[0].y:
        return "thumbs_down"
    return "unknown"

def process_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "❓ Unknown", "N/A"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        results = hands.process(img_rgb)
    annotated = img_rgb.copy()
    gesture_key = "unknown"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2)
            )
            gesture_key = classify_gesture(hand_landmarks.landmark)
        confidence = "High"
    else:
        confidence = "No hand detected"
    label = GESTURE_LABELS.get(gesture_key, "❓ Unknown")
    return annotated, label, confidence

# ── UI ────────────────────────────────────────────────────────
st.set_page_config(page_title="Hand Gesture Recognition", page_icon="🤚", layout="centered")

st.title("🤚 Hand Gesture Recognition")
st.write("Upload a hand image to detect the gesture using MediaPipe.")

uploaded = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    image_bytes = uploaded.read()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.image(Image.open(io.BytesIO(image_bytes)), use_container_width=True)
    with st.spinner("Detecting..."):
        annotated_np, gesture_label, confidence = process_image(image_bytes)
    with col2:
        st.markdown("**Landmarks**")
        if annotated_np is not None:
            st.image(annotated_np, use_container_width=True)
    st.success(f"Gesture: {gesture_label}  |  Confidence: {confidence}")
else:
    st.info("👆 Upload a hand image to get started.")