import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="🤚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
        background-color: #0d0d1a;
        color: #e0e0ff;
    }
    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #0d0d1a 100%);
    }
    h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; color: #00f5d4 !important; }
    .metric-card {
        background: rgba(0,245,212,0.05);
        border: 1px solid rgba(0,245,212,0.3);
        border-radius: 12px; padding: 20px; text-align: center; margin: 8px 0;
    }
    .gesture-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00f5d4, #7b2ff7);
        color: white; font-family: 'Orbitron', sans-serif;
        font-size: 1.4em; font-weight: 700;
        padding: 12px 28px; border-radius: 8px; letter-spacing: 2px;
        box-shadow: 0 0 24px rgba(0,245,212,0.4);
    }
    .info-box {
        background: rgba(123,47,247,0.1);
        border-left: 3px solid #7b2ff7;
        border-radius: 6px; padding: 14px 18px; margin: 10px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00f5d4, #7b2ff7);
        color: white; font-family: 'Orbitron', sans-serif;
        font-weight: 700; border: none; border-radius: 8px; padding: 10px 28px;
    }
</style>
""", unsafe_allow_html=True)

# ── MediaPipe setup (solutions API — requires mediapipe==0.10.9) ──────────────
mp_hands         = mp.solutions.hands
mp_drawing       = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ── Gesture Logic ─────────────────────────────────────────────────────────────

def get_finger_states(hand_landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    fingers = []

    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip  = hand_landmarks.landmark[3]
    wrist     = hand_landmarks.landmark[0]
    index_mcp = hand_landmarks.landmark[5]

    if wrist.x < index_mcp.x:
        fingers.append(thumb_tip.x > thumb_ip.x)
    else:
        fingers.append(thumb_tip.x < thumb_ip.x)

    for tip, pip in zip(tips, pips):
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)

    return fingers  # [Thumb, Index, Middle, Ring, Pinky]


def classify_gesture(fingers):
    thumb, index, middle, ring, pinky = fingers

    if not any(fingers):                                              return "FIST ✊",       "Closed fist."
    if all(fingers):                                                  return "OPEN HAND 🖐️", "All fingers extended."
    if not thumb and index and not middle and not ring and not pinky: return "POINTING ☝️",  "Index finger only."
    if not thumb and index and middle and not ring and not pinky:     return "PEACE ✌️",      "Victory sign."
    if thumb and index and not middle and not ring and not pinky:     return "L-SHAPE 👆",    "Thumb + index."
    if thumb and not index and not middle and not ring and pinky:     return "HANG LOOSE 🤙", "Thumb + pinky."
    if not thumb and not index and not middle and not ring and pinky: return "PINKY 🤙",      "Only pinky."
    if thumb and index and middle and not ring and not pinky:         return "THREE 🤟",      "Three fingers."
    if not thumb and index and middle and ring and pinky:             return "FOUR 🖖",       "Four fingers."
    if thumb and not index and not middle and not ring and not pinky: return "THUMBS UP 👍",  "Thumb up!"

    return f"CUSTOM ({sum(fingers)}) 🤚", f"{sum(fingers)} fingers detected."


def process_image(image_bgr):
    image_rgb   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results_out = []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                        min_detection_confidence=0.5) as hands:
        results = hands.process(image_rgb)

    annotated = image_bgr.copy()

    if results.multi_hand_landmarks:
        for idx, hand_lm in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                annotated, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            fingers       = get_finger_states(hand_lm)
            gesture, desc = classify_gesture(fingers)
            label         = results.multi_handedness[idx].classification[0].label
            conf          = results.multi_handedness[idx].classification[0].score
            results_out.append({"hand": label, "gesture": gesture,
                                 "desc": desc, "conf": conf, "fingers": fingers})

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), results_out


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤚 HGR Settings")
    st.markdown("---")
    show_landmarks = st.checkbox("Show Landmark Overlay", value=True)
    st.markdown("---")
    st.markdown("""
    <div class='info-box'>
    <b>Supported Gestures</b><br>
    ✊ Fist &nbsp; 🖐️ Open Hand<br>
    ☝️ Pointing &nbsp; ✌️ Peace<br>
    👍 Thumbs Up &nbsp; 🤙 Hang Loose<br>
    🤟 Three &nbsp; 🖖 Four
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("SkillCraft Technology · Task 04")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("# 🤚 Hand Gesture Recognition")
st.markdown("**Task 04 — SkillCraft Technology** | Powered by MediaPipe + Streamlit")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "🎥 Webcam", "ℹ️ About"])

# ── Tab 1: Image Upload ───────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Upload an Image")
        uploaded = st.file_uploader("Choose an image (JPG / PNG / WEBP)",
                                    type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            pil_img   = Image.open(uploaded).convert("RGB")
            image_np  = np.array(pil_img)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            st.image(pil_img, caption="Original Image", use_column_width=True)

    with col2:
        st.markdown("### Detection Result")
        if uploaded:
            with st.spinner("Detecting gestures…"):
                annotated_rgb, detections = process_image(image_bgr)

            st.image(annotated_rgb if show_landmarks else pil_img,
                     caption="Annotated Image", use_column_width=True)

            if detections:
                for d in detections:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='font-size:0.9em;color:#aaa;margin-bottom:6px'>
                            {d['hand']} Hand · Confidence: {d['conf']:.0%}
                        </div>
                        <div class='gesture-badge'>{d['gesture']}</div>
                        <div style='margin-top:10px;color:#c0c0e0'>{d['desc']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("**Finger States:**")
                    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                    fcols = st.columns(5)
                    for i, (fname, fstate) in enumerate(zip(finger_names, d["fingers"])):
                        with fcols[i]:
                            color  = "rgba(0,245,212,0.2)" if fstate else "rgba(255,60,60,0.1)"
                            border = "#00f5d4" if fstate else "#ff3c3c"
                            st.markdown(
                                f"<div style='text-align:center;padding:8px;border-radius:6px;"
                                f"background:{color};border:1px solid {border};font-size:0.8em'>"
                                f"{'✅' if fstate else '❌'}<br>{fname}</div>",
                                unsafe_allow_html=True)
            else:
                st.warning("⚠️ No hands detected. Try a clearer image with visible hands.")
        else:
            st.info("Upload an image to get started.")


# ── Tab 2: Webcam ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Live Webcam Gesture Detection")
    st.info("📷 Enable the checkbox below to start your webcam.")

    run_cam      = st.checkbox("▶ Start Camera")
    FRAME_WINDOW = st.empty()
    gesture_out  = st.empty()

    if run_cam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access webcam. Please allow camera permissions.")
        else:
            while run_cam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break
                annotated_rgb, detections = process_image(frame)
                FRAME_WINDOW.image(annotated_rgb, channels="RGB", use_column_width=True)
                if detections:
                    labels = " · ".join([d["gesture"] for d in detections])
                    gesture_out.markdown(
                        f"<div style='text-align:center'>"
                        f"<span class='gesture-badge'>{labels}</span></div>",
                        unsafe_allow_html=True)
                else:
                    gesture_out.markdown("*No hands detected*")
            cap.release()


# ── Tab 3: About ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### About This Project")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='info-box'>
        <b>Task 04 — SkillCraft Technology</b><br><br>
        Develop a hand gesture recognition model that accurately identifies
        and classifies different hand gestures from image or video data,
        enabling intuitive human-computer interaction.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        **Tech Stack**
        - 🖐️ **MediaPipe Hands** — landmark detection
        - 🐍 **Python + OpenCV** — image processing
        - 🌐 **Streamlit** — web interface
        """)
    with c2:
        st.markdown("""
        **How It Works**
        1. MediaPipe detects 21 hand landmarks per hand
        2. Finger states computed from Y-coordinates
        3. Rule-based classifier maps states → gesture label
        4. Results overlaid on the image/video frame

        | Finger | Tip | PIP |
        |--------|-----|-----|
        | Index  | 8   | 6   |
        | Middle | 12  | 10  |
        | Ring   | 16  | 14  |
        | Pinky  | 20  | 18  |
        | Thumb  | 4   | 3   |
        """)