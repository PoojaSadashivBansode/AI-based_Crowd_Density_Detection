import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import datetime
from models.yolo_detector import YoloDetector
from models.csrnet_estimator import CSRNetEstimator
import tempfile

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

# Sidebar Configuration
st.sidebar.title("🔧 Settings")
source_radio = st.sidebar.radio("Video Source", ["Sample Video", "Upload Video", "Webcam"])
threshold = st.sidebar.slider("⚠️ Density Threshold (Alert)", 10, 500, 50)
model_select = st.sidebar.selectbox("Model Preference", ["Auto (Hybrid)", "YOLOv8 Only", "CSRNet Only"])
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
            
            # Model Switching Logic:
            # - YOLO: Good for sparse/medium crowds (< 20 people, minimal overlapping)
            # - CSRNet: Good for dense crowds (>= 20 people, overlapping people)
            # Note: This is independent of the alert threshold!
            
            density_switch_threshold = 20  # Switch to CSRNet when crowd gets dense
            
            if model_select == "CSRNet Only":
                # Force CSRNet mode
                c_count, density_map = csrnet_model.estimate(frame)
                calibration_factor = 25.0
                count = int(abs(c_count) * calibration_factor)
                mode = "CSRNet"
                
                # Visualize Density Map
                density_map_norm = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
                density_map_color = cv2.applyColorMap((density_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                density_map_color = cv2.resize(density_map_color, (frame.shape[1], frame.shape[0]))
                annotated_frame = cv2.addWeighted(frame, 0.6, density_map_color, 0.4, 0)
                
            elif model_select == "Auto (Hybrid)" and count >= density_switch_threshold:
                # Auto mode: Switch to CSRNet when density is high (people overlapping)
                c_count, density_map = csrnet_model.estimate(frame)
                calibration_factor = 25.0
                count = int(abs(c_count) * calibration_factor)
                mode = "CSRNet"
                
                # Visualize Density Map
                density_map_norm = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
                density_map_color = cv2.applyColorMap((density_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                density_map_color = cv2.resize(density_map_color, (frame.shape[1], frame.shape[0]))
                annotated_frame = cv2.addWeighted(frame, 0.6, density_map_color, 0.4, 0)
            
            # If YOLOv8 Only or count < density_switch_threshold, stay in YOLO mode
            
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

            # 3. Alert System
            if count >= threshold:
                alert_placeholder.markdown(f"<div class='alert-box'>🚨 ALERT: CROWD LIMIT EXCEEDED ({count} > {threshold})</div>", unsafe_allow_html=True)
                # Overlay on video
                cv2.putText(annotated_frame, f"ALERT: {count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            else:
                alert_placeholder.empty()

            # 4. Video Display
            # Convert color space only when displaying
            video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

            # 5. Graph Update
            # Append new data
            now = datetime.datetime.now()
            new_row = pd.DataFrame({'Time': [now], 'Count': [count]})
            df_log = pd.concat([df_log, new_row], ignore_index=True)
            
            # Keep last 100 points for performance
            if len(df_log) > 100:
                df_log = df_log.iloc[-100:]
            
            # Use Streamlit's native line chart
            chart_placeholder.line_chart(df_log.set_index('Time'))
            
            # Stop button logic handled by Streamlit rerun implicitly on logic change
