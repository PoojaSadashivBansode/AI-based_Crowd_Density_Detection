import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import datetime
from models.yolo_detector import YoloDetector
from models.csrnet_estimator import CSRNetEstimator
import tempfile
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
    duration = 0.5  # seconds
    frequency = 800  # Hz (beep tone)
    
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Create repeating beeps (3 beeps)
    silence = np.zeros(int(sample_rate * 0.2))
    alarm = np.concatenate([audio_data, silence, audio_data, silence, audio_data])
    
    # Convert to 16-bit PCM
    alarm = (alarm * 32767).astype(np.int16)
    
    # Create WAV file in memory
    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(alarm.tobytes())
    
    buffer.seek(0)
    return buffer.getvalue()

# Multi-Factor Model Switching Helper Functions
def calculate_box_overlap_ratio(boxes):
    """
    Calculate the overlap ratio between bounding boxes.
    High overlap indicates dense crowd where YOLO becomes unreliable.
    """
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
            
            # Calculate intersection
            x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            overlap_area = x_overlap * y_overlap
            total_overlap += overlap_area
    
    # Return overlap ratio (0 to 1)
    if total_area == 0:
        return 0.0
    return min(1.0, total_overlap / total_area)

def calculate_crowd_coverage(boxes, frame_shape):
    """
    Calculate what percentage of the frame is occupied by detected people.
    High coverage indicates dense crowd.
    """
    if len(boxes) == 0:
        return 0.0
    
    frame_area = frame_shape[0] * frame_shape[1]
    total_box_area = 0
    
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        box_area = (x_max - x_min) * (y_max - y_min)
        total_box_area += box_area
    
    coverage = total_box_area / frame_area
    return min(1.0, coverage)  # Cap at 100%

def should_switch_to_csrnet(count, boxes, frame_shape, prev_counts, count_threshold=30):
    """
    Multi-factor decision function to switch from YOLO to CSRNet.
    
    Factors:
    1. YOLO count >= threshold (30)
    2. High bounding box overlap (>= 0.15)
    3. High crowd coverage area (>= 0.25)
    4. Sudden drop in count (potential occlusion)
    
    Returns:
        bool: True if should switch to CSRNet
        str: Reason for switching
    """
    reasons = []
    
    # Factor 1: Count threshold
    if count >= count_threshold:
        reasons.append(f"High count ({count} ≥ {count_threshold})")
    
    # Factor 2: Bounding box overlap
    overlap_ratio = calculate_box_overlap_ratio(boxes)
    if overlap_ratio >= 0.15:  # 15% overlap threshold
        reasons.append(f"High overlap ({overlap_ratio:.2%})")
    
    # Factor 3: Crowd coverage
    coverage = calculate_crowd_coverage(boxes, frame_shape)
    if coverage >= 0.25:  # 25% frame coverage
        reasons.append(f"High coverage ({coverage:.2%})")
    
    # Factor 4: Sudden drop detection (if we have history)
    if len(prev_counts) >= 3:
        avg_recent = np.mean(prev_counts[-3:])
        if avg_recent > count_threshold and count < avg_recent * 0.7:
            reasons.append(f"Sudden drop ({count} < {avg_recent:.0f})")
    
    # Switch if at least 2 factors are triggered
    should_switch = len(reasons) >= 2
    reason_str = ", ".join(reasons) if reasons else "None"
    
    return should_switch, reason_str

# Sidebar Configuration
st.sidebar.title("🔧 Settings")
source_radio = st.sidebar.radio("Video Source", ["Sample Video", "Upload Video", "Webcam"])
threshold = st.sidebar.number_input("⚠️ Density Threshold (Alert)", min_value=10, max_value=500, value=50, step=5)
model_select = st.sidebar.selectbox("Model Preference", ["Auto (Hybrid)", "YOLOv8 Only", "CSRNet Only"])
enable_alarm = st.sidebar.checkbox("🔔 Enable Alarm Sound", value=True)
run_app = st.sidebar.button("🚀 Start Monitoring")

# Initialize Models (Cached)
@st.cache_resource
def load_models():
    yolo = YoloDetector('yolov8s.pt') # Using Small model as requested
    try:
        csrnet = CSRNetEstimator('csrnet_weights.pth')
    except:
        st.sidebar.warning("CSRNet weights not found. Using initialized weights.")
        csrnet = CSRNetEstimator(None)
    return yolo, csrnet

yolo_model, csrnet_model = load_models()

# Main Layout
st.title("🛡️ AI-Based Crowd Density Control")

# Top Metrics Row (Placeholders)
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
    info_placeholder = st.empty()

with col_graph:
    st.subheader("📈 Crowd Trend")
    chart_placeholder = st.empty()
    
# Logic
if run_app:
    # Video Source Logic
    cap = None
    if source_radio == "Sample Video":
        cap = cv2.VideoCapture("video.mp4")
    elif source_radio == "Upload Video":
        uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
    elif source_radio == "Webcam":
        cap = cv2.VideoCapture(0)

    if cap is None or not cap.isOpened():
        st.error("Error loading video source.")
    else:
        # Loop Variables
        frame_window = []
        df_log = pd.DataFrame(columns=['Time', 'Count'])
        peak_count = 0
        prev_time = time.time()
        count_history = []  # Track count history for sudden drop detection
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Loop video if sample
                if source_radio == "Sample Video":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            
            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # 1. Processing
            count = 0
            annotated_frame = frame.copy()
            use_csrnet = False
            
            # Hybrid Logic
            if model_select == "Auto (Hybrid)":
                # Simple logic: Run YOLO first. If close to threshold, maybe switch?
                # For efficiency in this demo, let's use the threshold strategy
                # Note: To know to switch to CSRNet without running YOLO, we need history.
                # Here we will do: If last count > threshold, try CSRNet.
                pass 
                
            # For simplicity in Streamlit loop, let's stick to the logic:
            # We need a quick estimate. YOLO is good default.
            
            # Decide model
            # To avoid oscillation, we can use a simple state check if we tracked state.
            # Here we default to YOLO unless forced or high count derived from YOLO
            
            # YOLO Detection (always run first for initial count)
            count, boxes, annotated_frame = yolo_model.detect(frame)
            mode = "YOLO"
            switch_reason = "N/A"
            
            # Multi-Factor Model Switching Logic:
            # Factor 1: YOLO count threshold (≥ 30)
            # Factor 2: Bounding box overlap (dense overlapping people)
            # Factor 3: Crowd coverage area (large portion of frame occupied)
            # Factor 4: Sudden drop in YOLO count (occlusion detected)
            # Switch to CSRNet if at least 2 factors are triggered
            
            if model_select == "CSRNet Only":
                # Force CSRNet mode
                c_count, density_map = csrnet_model.estimate(frame)
                calibration_factor = 0.18  # Fixed: was 20.0 (100x too high)
                count = int(abs(c_count) * calibration_factor)
                mode = "CSRNet (Forced)"
                switch_reason = "Manual selection"
                
                # Visualize Density Map
                density_map_norm = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
                density_map_color = cv2.applyColorMap((density_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                density_map_color = cv2.resize(density_map_color, (frame.shape[1], frame.shape[0]))
                annotated_frame = cv2.addWeighted(frame, 0.6, density_map_color, 0.4, 0)
                
            elif model_select == "Auto (Hybrid)":
                # Intelligent auto-switching based on multiple factors
                should_switch, switch_reason = should_switch_to_csrnet(
                    count, boxes, frame.shape, count_history, count_threshold=30
                )
                
                if should_switch:
                    # Switch to CSRNet for dense crowd estimation
                    c_count, density_map = csrnet_model.estimate(frame)
                    calibration_factor = 0.18  # Fixed: was 25.0 (100x too high)
                    count = int(abs(c_count) * calibration_factor)
                    mode = "CSRNet (Auto)"
                    
                    # Visualize Density Map
                    density_map_norm = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
                    density_map_color = cv2.applyColorMap((density_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    density_map_color = cv2.resize(density_map_color, (frame.shape[1], frame.shape[0]))
                    annotated_frame = cv2.addWeighted(frame, 0.6, density_map_color, 0.4, 0)
                else:
                    switch_reason = "Sparse crowd detected"
            
            # Update count history (keep last 10 frames)
            count_history.append(count)
            if len(count_history) > 10:
                count_history.pop(0)
            
            # If YOLOv8 Only, stay in YOLO mode
            
            # 2. Metrics Update
            if count > peak_count:
                peak_count = count
            
            # Status
            status_text = "🟢 SAFE"
            status_color = "#00FF00" # Green
            if count >= threshold:
                status_text = "🔴 ALERT"
                status_color = "#FF0000" # Red
                
            # Render KPIs (Using HTML for nice formatting)
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
                <div class='metric-label'>🏆 Peak Today</div>
                <div class='metric-value' style='font-size: 28px;'>{peak_count}</div>
            </div>
            """, unsafe_allow_html=True)

            kpi_fps.metric("⚡ Processing Speed", f"{int(fps)} FPS")

            # Display Active Model Info (overlay on frame)
            model_color = (0, 255, 255) if "CSRNet" in mode else (255, 150, 0)  # Yellow for CSRNet, Orange for YOLO
            cv2.putText(annotated_frame, f"Model: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_color, 2)
            if switch_reason and switch_reason != "N/A":
                cv2.putText(annotated_frame, f"Reason: {switch_reason}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 3. Alert System
            if count >= threshold:
                alert_placeholder.markdown(f"<div class='alert-box'>🚨 ALERT: CROWD LIMIT EXCEEDED ({count} > {threshold})</div>", unsafe_allow_html=True)
                # Overlay on video
                cv2.putText(annotated_frame, f"ALERT: {count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                
                # Play alarm sound if enabled
                if enable_alarm:
                    alarm_audio = generate_alarm_sound()
                    audio_base64 = base64.b64encode(alarm_audio).decode()
                    audio_html = f"""
                        <audio autoplay>
                            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                        </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
            else:
                alert_placeholder.empty()

            # 4. Video Display
            # Convert color space only when displaying
            video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            info_placeholder.info(f"📊 Current Mode: {mode} | Reason: {switch_reason}")
            
            # 5. Graph Update
            # Append new data
            now = datetime.datetime.now()
            new_row = pd.DataFrame({'Time': [now], 'Count': [count]})
            df_log = pd.concat([df_log, new_row], ignore_index=True)
            
            # Keep last 50 points for performance
            if len(df_log) > 50:
                df_log = df_log.iloc[-50:]
            
            # Use Streamlit's native line chart
            chart_placeholder.line_chart(df_log.set_index('Time'))
            
            # Stop button logic handled by Streamlit rerun implicitly on logic change

            # Control playback speed (adjust for smoother video)
            time.sleep(0.03)  # ~30 FPS equivalent
        cap.release()