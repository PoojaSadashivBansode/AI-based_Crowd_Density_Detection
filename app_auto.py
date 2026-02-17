import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import datetime
from models.yolo_detector import YoloDetector
from models.csrnet_estimator import CSRNetEstimator
import base64
from io import BytesIO
import wave
import os

# Page Config
st.set_page_config(
    page_title="Crowd Control Dashboard - Auto",
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

# Helper Functions
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
            overlap_area = x_overlap * y_overlap
            total_overlap += overlap_area
    
    if total_area == 0:
        return 0.0
    return min(1.0, total_overlap / total_area)

def calculate_crowd_coverage(boxes, frame_shape):
    if len(boxes) == 0:
        return 0.0
    
    frame_area = frame_shape[0] * frame_shape[1]
    total_box_area = 0
    
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        box_area = (x_max - x_min) * (y_max - y_min)
        total_box_area += box_area
    
    coverage = total_box_area / frame_area
    return min(1.0, coverage)

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
    reason_str = ", ".join(reasons) if reasons else "None"
    
    return should_switch, reason_str

# Check for video file
if not os.path.exists('video.mp4'):
    st.error("❌ video.mp4 not found! Please upload video.mp4 to the Colab Files sidebar.")
    st.stop()

# Sidebar Configuration
st.sidebar.title("🔧 Settings")
st.sidebar.info("📹 Auto-playing video.mp4")
threshold = st.sidebar.number_input("⚠️ Density Threshold (Alert)", min_value=10, max_value=500, value=50, step=5)
model_select = st.sidebar.selectbox("Model Preference", ["Auto (Hybrid)", "YOLOv8 Only", "CSRNet Only"])
enable_alarm = st.sidebar.checkbox("🔔 Enable Alarm Sound", value=False)

# Initialize Models (Cached)
@st.cache_resource
def load_models():
    yolo = YoloDetector('yolov8s.pt')
    try:
        csrnet = CSRNetEstimator('csrnet_weights.pth')
    except:
        st.sidebar.warning("CSRNet weights not found. Using initialized weights.")
        csrnet = CSRNetEstimator(None)
    return yolo, csrnet

yolo_model, csrnet_model = load_models()

# Main Layout
st.title("🛡️ AI-Based Crowd Density Control (Auto-Start)")

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    kpi_count = st.empty()
with col2:
    kpi_status = st.empty()
with col3:
    kpi_peak = st.empty()
with col4:
    kpi_fps = st.empty()

# Content Row
col_video, col_graph = st.columns([1.5, 1])

with col_video:
    st.subheader("🟢 Live Video Feed")
    video_placeholder = st.empty()
    alert_placeholder = st.empty()

with col_graph:
    st.subheader("📈 Crowd Trend")
    chart_placeholder = st.empty()

# AUTO-START (No button needed)
cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    st.error("Error loading video.mp4")
    st.stop()

# Initialize session state
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
    st.session_state.count_history = []
    st.session_state.peak_count = 0
    st.session_state.df_log = pd.DataFrame(columns=['Time', 'Count'])

# Process one frame per rerun
ret, frame = cap.read()
if not ret:
    # Loop video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        st.error("Cannot read video")
        st.stop()

st.session_state.frame_count += 1

# Processing
annotated_frame = frame.copy()
mode = "YOLO"
switch_reason = "N/A"

if model_select == "CSRNet Only":
    c_count, density_map = csrnet_model.estimate(frame)
    calibration_factor = 0.18
    count = int(abs(c_count) * calibration_factor)
    mode = "CSRNet (Forced)"
    switch_reason = "Manual selection"
    
    density_map_norm = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
    density_map_color = cv2.applyColorMap((density_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    density_map_color = cv2.resize(density_map_color, (frame.shape[1], frame.shape[0]))
    annotated_frame = cv2.addWeighted(frame, 0.6, density_map_color, 0.4, 0)
    
elif model_select == "Auto (Hybrid)":
    count, boxes, annotated_frame = yolo_model.detect(frame)
    
    should_switch, switch_reason = should_switch_to_csrnet(
        count, boxes, frame.shape, st.session_state.count_history, count_threshold=30
    )
    
    if should_switch:
        c_count, density_map = csrnet_model.estimate(frame)
        calibration_factor = 0.18
        count = int(abs(c_count) * calibration_factor)
        mode = "CSRNet (Auto)"
        
        density_map_norm = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
        density_map_color = cv2.applyColorMap((density_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        density_map_color = cv2.resize(density_map_color, (frame.shape[1], frame.shape[0]))
        annotated_frame = cv2.addWeighted(frame, 0.6, density_map_color, 0.4, 0)
    else:
        switch_reason = "Sparse crowd detected"
else:
    count, boxes, annotated_frame = yolo_model.detect(frame)
    mode = "YOLO"
    switch_reason = "Manual selection"

# Update history
st.session_state.count_history.append(count)
if len(st.session_state.count_history) > 10:
    st.session_state.count_history.pop(0)

# Update peak
if count > st.session_state.peak_count:
    st.session_state.peak_count = count

# Status
status_text = "🟢 SAFE"
status_color = "#00FF00"
if count >= threshold:
    status_text = "🔴 ALERT"
    status_color = "#FF0000"

# Render KPIs
kpi_count.markdown(f"""
<div class='metric-card'>
    <div class='metric-label'>👥 Current Crowd</div>
    <div class='metric-value'>{count}</div>
</div>
""", unsafe_allow_html=True)

kpi_status.markdown(f"""
<div class='metric-card' style='border-color: {status_color};'>
    <div class='metric-label'>🚦 Status</div>
    <div class='metric-value' style='color: {status_color}; font-size: 24px;'>{status_text}</div>
</div>
""", unsafe_allow_html=True)

kpi_peak.markdown(f"""
<div class='metric-card'>
    <div class='metric-label'>🏆 Peak</div>
    <div class='metric-value' style='font-size: 28px;'>{st.session_state.peak_count}</div>
</div>
""", unsafe_allow_html=True)

kpi_fps.metric("📊 Frame", f"{st.session_state.frame_count}")

# Display Model Info
model_color = (0, 255, 255) if "CSRNet" in mode else (255, 150, 0)
cv2.putText(annotated_frame, f"Model: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_color, 2)
if switch_reason and switch_reason != "N/A":
    cv2.putText(annotated_frame, f"Reason: {switch_reason}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# Alert
if count >= threshold:
    alert_placeholder.markdown(f"<div class='alert-box'>🚨 ALERT: CROWD LIMIT EXCEEDED ({count} > {threshold})</div>", unsafe_allow_html=True)
    cv2.putText(annotated_frame, f"ALERT: {count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
else:
    alert_placeholder.empty()

# Video Display
video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

# Graph Update
now = datetime.datetime.now()
new_row = pd.DataFrame({'Time': [now], 'Count': [count]})
st.session_state.df_log = pd.concat([st.session_state.df_log, new_row], ignore_index=True)

if len(st.session_state.df_log) > 50:
    st.session_state.df_log = st.session_state.df_log.iloc[-50:]

chart_placeholder.line_chart(st.session_state.df_log.set_index('Time'))

cap.release()

# Auto-rerun for next frame
time.sleep(0.1)
st.rerun()
