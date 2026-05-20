import streamlit as st
from PIL import Image
import numpy as np
import io
import cv2

def detect_gesture(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Skin color range
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(img_hsv, lower, upper)
    
    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img_rgb, "No hand detected", "Low"
    
    # Largest contour = hand
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    
    if area < 3000:
        return img_rgb, "No hand detected", "Low"
    
    # Draw contour
    annotated = img_rgb.copy()
    cv2.drawContours(annotated, [contour], -1, (0, 255, 100), 3)
    
    # Convex hull + defects to count fingers
    hull = cv2.convexHull(contour, returnPoints=False)
    
    try:
        defects = cv2.convexityDefects(contour, hull)
    except:
        defects = None
    
    fingers = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # Angle at the defect point
            a = np.linalg.norm(np.array(end) - np.array(start))
            b = np.linalg.norm(np.array(far) - np.array(start))
            c = np.linalg.norm(np.array(end) - np.array(far))
            
            if 2 * b * c == 0:
                continue
            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
            
            if angle <= np.pi / 2 and d > 10000:
                fingers += 1
                cv2.circle(annotated, far, 8, (255, 0, 0), -1)
    
    fingers = min(fingers + 1, 5)
    
    gesture_map = {
        1: "☝️ Pointing / 👍 Thumbs Up",
        2: "✌️ Peace",
        3: "🤟 Three Fingers",
        4: "🖐 Four Fingers",
        5: "✋ Open Hand"
    }
    
    if fingers == 0:
        gesture = "✊ Fist"
    else:
        gesture = gesture_map.get(fingers, "✋ Open Hand")
    
    return annotated, gesture, "Medium"

# ── UI ──────────────────────────────────────────────────────
st.set_page_config(page_title="Hand Gesture Recognition", page_icon="🤚", layout="centered")

st.title("🤚 Hand Gesture Recognition")
st.write("Upload a hand image to detect the gesture.")

uploaded = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    image_bytes = uploaded.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.image(img, use_container_width=True)
    
    with st.spinner("Detecting..."):
        annotated, gesture, confidence = detect_gesture(img_np)
    
    with col2:
        st.markdown("**Detection**")
        st.image(annotated, use_container_width=True)
    
    st.success(f"Gesture: {gesture}  |  Confidence: {confidence}")

else:
    st.info("👆 Upload a clear hand image on a plain background.")

st.markdown("---")
st.markdown("<center style='color:gray;font-size:0.8rem;'>SkillCraft Technology · Task 04</center>", unsafe_allow_html=True)