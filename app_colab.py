import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import datetime
import os
import tempfile
from models.yolo_detector import YoloDetector
from models.csrnet_estimator import CSRNetEstimator
import base64
from io import BytesIO
import wave

# Page Config
st.set_page_config(
    page_title="Crowd Control Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #00ADB5;
    }
    .metric-label {
        font-size: 14px;
        color: #AAAAAA;
    }
    .alert-box {
        background-color: #FF4B4B;
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        animation: blinking 1s infinite;
    }
    @keyframes blinking {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

# Generate Alarm Sound
def generate_alarm_sound():
    """Generate a simple alarm beep sound"""
    sample_rate = 44100
    duration = 0.5
    frequency = 800

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t)

    silence = np.zeros(int(sample_rate * 0.2))
    alarm = np.concatenate([audio_data, silence, audio_data, silence, audio_data])
    alarm = (alarm * 32767).astype(np.int16)

    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(alarm.tobytes())

    buffer.seek(0)
    return buffer.getvalue()

# Multi-Factor Model Switching Helper Functions
def calculate_box_overlap_ratio(boxes):
    if len(boxes) < 2:
        return 0.0
    total_overlap = 0
    total_area = 0
    for i, box1 in enumerate(boxes):
        x1_min, y1_min, x1_max, y1_max = box1
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        total_area += area1
        for box2 in boxes[i+1:]:
            x2_min, y2_min, x2_max, y2_max = box2
            x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            total_overlap += x_overlap * y_overlap
    if total_area == 0:
        return 0.0
    return min(1.0, total_overlap / total_area)

def calculate_crowd_coverage(boxes, frame_shape):
    if len(boxes) == 0:
        return 0.0
    frame_area = frame_shape[0] * frame_shape[1]
    total_box_area = sum((x_max - x_min) * (y_max - y_min) for x_min, y_min, x_max, y_max in boxes)
    return min(1.0, total_box_area / frame_area)

def should_switch_to_csrnet(count, boxes, frame_shape, prev_counts, count_threshold=30):
    reasons = []
    if count >= count_threshold:
        reasons.append(f"High count ({count} >= {count_threshold})")
    overlap_ratio = calculate_box_overlap_ratio(boxes)
    if overlap_ratio >= 0.15:
        reasons.append(f"High overlap ({overlap_ratio:.2%})")
    coverage = calculate_crowd_coverage(boxes, frame_shape)
    if coverage >= 0.25:
        reasons.append(f"High coverage ({coverage:.2%})")
    if len(prev_counts) >= 3:
        avg_recent = np.mean(prev_counts[-3:])
        if avg_recent > count_threshold and count < avg_recent * 0.7:
            reasons.append(f"Sudden drop ({count} < {avg_recent:.0f})")
    should_switch = len(reasons) >= 2
    return should_switch, (", ".join(reasons) if reasons else "None")

# ──────────────────────────────────────────────
# Sidebar Configuration
# ──────────────────────────────────────────────
st.sidebar.title("🔧 Settings")
source_radio = st.sidebar.radio("Video Source", ["Sample Video", "Upload Video"])
# Note: Webcam is not available on Colab — use Sample or Upload

# File uploader MUST be outside run block so it persists across Streamlit reruns
if source_radio == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("📂 Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        st.session_state["uploaded_video_path"] = tfile.name
    elif "uploaded_video_path" not in st.session_state:
        st.sidebar.info("Please upload a video file first.")

threshold = st.sidebar.number_input("⚠️ Density Threshold (Alert)", min_value=10, max_value=500, value=50, step=5)
model_select = st.sidebar.selectbox("Model Preference", ["Auto (Hybrid)", "YOLOv8 Only", "CSRNet Only"])
enable_alarm = st.sidebar.checkbox("🔔 Enable Alarm Sound", value=False)  # Off by default on Colab

col_start, col_stop = st.sidebar.columns(2)
run_app  = col_start.button("🚀 Start", use_container_width=True)
stop_app = col_stop.button("⏹ Stop",  use_container_width=True)

if stop_app:
    st.session_state["running"] = False
if run_app:
    st.session_state["running"] = True

# CSRNet calibration: official ShanghaiTech Part A weights overcount on normal scenes
csrnet_scale = 0.20

# ──────────────────────────────────────────────
# Initialize Models (Cached)
# ──────────────────────────────────────────────
import torch

@st.cache_resource
def load_models():
    yolo = YoloDetector('yolov8s.pt')
    try:
        csrnet = CSRNetEstimator('csrnet_weights.pth')
    except:
        st.sidebar.warning("CSRNet weights not found. Using initialized weights.")
        csrnet = CSRNetEstimator(None)
    device = "GPU 🚀" if torch.cuda.is_available() else "CPU 💻"
    st.sidebar.success(f"Running on: {device}")
    return yolo, csrnet

yolo_model, csrnet_model = load_models()

# ──────────────────────────────────────────────
# Main Layout
# ──────────────────────────────────────────────
st.title("🛡️ AI-Based Crowd Density Control")

col1, col2, col3, col4 = st.columns(4)
with col1: kpi_count  = st.empty()
with col2: kpi_status = st.empty()
with col3: kpi_peak   = st.empty()
with col4: kpi_fps    = st.empty()

col_video, col_graph = st.columns([1.5, 1])
with col_video:
    st.subheader("🟢 Live Video Feed")
    video_placeholder = st.empty()
    alert_placeholder = st.empty()
with col_graph:
    st.subheader("📈 Crowd Trend")
    chart_placeholder = st.empty()

# ──────────────────────────────────────────────
# Main Processing Loop
# ──────────────────────────────────────────────
if st.session_state.get("running", False):
    # Video Source
    cap = None
    if source_radio == "Sample Video":
        if not os.path.exists('video.mp4'):
            st.error("❌ video.mp4 not found! Upload it to the Colab Files sidebar.")
            st.stop()
        cap = cv2.VideoCapture("video.mp4")
    elif source_radio == "Upload Video":
        video_path = st.session_state.get("uploaded_video_path")
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            st.warning("Please upload a video file and click Start again.")

    if cap is None or not cap.isOpened():
        st.error("Error loading video source.")
    else:
        df_log = pd.DataFrame(columns=['Time', 'Count'])
        peak_count = 0
        prev_time = time.time()
        count_history = []

        while cap.isOpened():
            # Respect Stop button
            if not st.session_state.get("running", False):
                cap.release()
                break

            ret, frame = cap.read()
            if not ret:
                # Loop video at end
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 0.001)
            prev_time = curr_time

            # ── Model Inference ──
            annotated_frame = frame.copy()
            mode = "YOLO"
            switch_reason = "N/A"

            # Always run YOLO first
            count, boxes, annotated_frame = yolo_model.detect(frame)

            if model_select == "CSRNet Only":
                c_count, density_map = csrnet_model.estimate(frame)
                count = int(c_count * csrnet_scale)
                mode = "CSRNet (Forced)"
                switch_reason = "Manual selection"
                density_map_norm = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-7)
                density_map_color = cv2.applyColorMap((density_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                density_map_color = cv2.resize(density_map_color, (frame.shape[1], frame.shape[0]))
                annotated_frame = cv2.addWeighted(frame, 0.6, density_map_color, 0.4, 0).astype(np.uint8)

            elif model_select == "Auto (Hybrid)":
                should_switch, switch_reason = should_switch_to_csrnet(
                    count, boxes, frame.shape, count_history, count_threshold=30
                )
                if should_switch:
                    c_count, density_map = csrnet_model.estimate(frame)
                    count = int(c_count * csrnet_scale)
                    mode = "CSRNet (Auto)"
                    density_map_norm = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-7)
                    density_map_color = cv2.applyColorMap((density_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    density_map_color = cv2.resize(density_map_color, (frame.shape[1], frame.shape[0]))
                    annotated_frame = cv2.addWeighted(frame, 0.6, density_map_color, 0.4, 0).astype(np.uint8)
                else:
                    switch_reason = "Sparse crowd detected"

            # Count history
            count_history.append(count)
            if len(count_history) > 10:
                count_history.pop(0)

            # ── Metrics ──
            if count > peak_count:
                peak_count = count

            status_text  = "🟢 SAFE"
            status_color = "#00FF00"
            if count >= threshold:
                status_text  = "🔴 ALERT"
                status_color = "#FF0000"

            kpi_count.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>👥 Current Crowd</div>
                <div class='metric-value'>{count}</div>
            </div>""", unsafe_allow_html=True)

            kpi_status.markdown(f"""
            <div class='metric-card' style='border-color: {status_color};'>
                <div class='metric-label'>🚦 Status</div>
                <div class='metric-value' style='color: {status_color}; font-size: 24px;'>{status_text}</div>
            </div>""", unsafe_allow_html=True)

            kpi_peak.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>🏆 Peak Today</div>
                <div class='metric-value' style='font-size: 28px;'>{peak_count}</div>
            </div>""", unsafe_allow_html=True)

            kpi_fps.metric("⚡ Processing Speed", f"{int(fps)} FPS")

            # ── Overlay on frame ──
            model_color = (0, 255, 255) if "CSRNet" in mode else (255, 150, 0)
            cv2.putText(annotated_frame, f"Model: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_color, 2)
            if switch_reason and switch_reason != "N/A":
                cv2.putText(annotated_frame, f"Reason: {switch_reason}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # ── Alert ──
            if count >= threshold:
                alert_placeholder.markdown(
                    f"<div class='alert-box'>🚨 ALERT: CROWD LIMIT EXCEEDED ({count} > {threshold})</div>",
                    unsafe_allow_html=True)
                cv2.putText(annotated_frame, f"ALERT: {count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                if enable_alarm:
                    alarm_audio = generate_alarm_sound()
                    audio_base64 = base64.b64encode(alarm_audio).decode()
                    st.markdown(f"""<audio autoplay>
                        <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                    </audio>""", unsafe_allow_html=True)
            else:
                alert_placeholder.empty()

            # ── Video Display ──
            try:
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
                video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
            except Exception as e:
                st.error(f"Render Error: {e}")

            # ── Chart ──
            now = datetime.datetime.now()
            new_row = pd.DataFrame({'Time': [now], 'Count': [count]})
            df_log = pd.concat([df_log, new_row], ignore_index=True)
            if len(df_log) > 100:
                df_log = df_log.iloc[-100:]
            chart_placeholder.line_chart(df_log.set_index('Time'))

            time.sleep(0.05)  # Colab: small delay to keep UI responsive
