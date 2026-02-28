import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime
import tempfile
import os

from models.yolo_detector import YoloDetector
from models.csrnet_estimator import CSRNetEstimator
from utils.alert_system import AlertSystem

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Crowd Density Monitor", layout="wide")
st.title("🚦 AI-Based Crowd Density Prediction System")

# -------------------------------
# SIDEBAR SETTINGS
# -------------------------------
st.sidebar.header("⚙ Settings")

threshold = st.sidebar.slider("Alert Threshold", 10, 300, 50)
yolo_weights = st.sidebar.selectbox("YOLO Model",
                                    ["yolov8s.pt", "yolov8m.pt", "yolov8l.pt"])

use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=True)

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload Crowd Video", type=["mp4", "avi", "mov"])

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def compute_density_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])


def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = ((x1_max - x1_min) * (y1_max - y1_min) +
             (x2_max - x2_min) * (y2_max - y2_min) - inter_area + 1e-6)

    return inter_area / union


def compute_overlap_ratio(boxes):
    if len(boxes) < 2:
        return 0.0

    overlaps = 0
    total = len(boxes)

    for i in range(total):
        for j in range(i + 1, total):
            if compute_iou(boxes[i], boxes[j]) > 0.3:
                overlaps += 1

    return overlaps / total


# -------------------------------
# PROCESS BUTTON
# -------------------------------
if uploaded_file is not None:

    # Save temp video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.success("✅ Video uploaded successfully!")

    if st.button("🚀 Start Processing"):

        st.info("Initializing models...")

        # Load models
        yolo_model = YoloDetector(yolo_weights)
        csrnet_model = CSRNetEstimator("csrnet_weights.pth")
        alert_system = AlertSystem(threshold)

        cap = cv2.VideoCapture(video_path)

        frame_placeholder = st.empty()
        stats_placeholder = st.empty()

        log_data = []

        prev_yolo_count = 0
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # ---------------- YOLO DETECTION ----------------
            yolo_count, boxes, annotated_frame = yolo_model.detect(frame)

            # average confidence
            confidences = []
            for b in boxes:
                if len(b) == 5:
                    confidences.append(b[4])
            avg_conf = np.mean(confidences) if confidences else 0.5

            # ---------------- METRICS ----------------
            density_score = compute_density_score(frame)
            overlap_ratio = compute_overlap_ratio(boxes)

            # ---------------- DECISION LOGIC ----------------
            USE_CSRNET = False
            reason = "Sparse crowd"

            if yolo_count < 35 and density_score > 0.30:
                USE_CSRNET = True
                reason = "Dense crowd (low YOLO + high density)"

            elif overlap_ratio > 0.30:
                USE_CSRNET = True
                reason = "High overlap"

            elif avg_conf < 0.45:
                USE_CSRNET = True
                reason = "Low YOLO confidence"

            elif abs(yolo_count - prev_yolo_count) > 15:
                USE_CSRNET = True
                reason = "Unstable YOLO count"

            # ---------------- FINAL COUNT ----------------
            if USE_CSRNET:
                csr_count, density_map = csrnet_model.estimate(frame)
                final_count = int(csr_count)
                mode = "CSRNet"

                density_map_norm = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
                density_map_color = cv2.applyColorMap((density_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                density_map_color = cv2.resize(density_map_color, (frame.shape[1], frame.shape[0]))

                annotated_frame = cv2.addWeighted(frame, 0.6, density_map_color, 0.4, 0)

            else:
                final_count = yolo_count
                mode = "YOLO"

            prev_yolo_count = yolo_count

            # ---------------- ALERT ----------------
            triggered, _ = alert_system.check_alert(final_count)

            # ---------------- DRAW TEXT ----------------
            color = (0, 255, 255) if mode == "CSRNet" else (255, 150, 0)

            cv2.putText(annotated_frame, f"Mode: {mode}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.putText(annotated_frame, f"Count: {final_count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.putText(annotated_frame, f"Reason: {reason}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if triggered:
                cv2.putText(annotated_frame, "🚨 ALERT HIGH DENSITY",
                            (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3)

            # ---------------- STREAMLIT DISPLAY (FIXED) ----------------
            frame_placeholder.image(annotated_frame, channels="BGR")

            stats_placeholder.write({
                "Frame": frame_id,
                "Mode": mode,
                "Count": final_count,
                "Reason": reason
            })

            # ---------------- LOG ----------------
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            log_data.append([timestamp, frame_id, mode, final_count, reason])

        cap.release()

        # ---------------- SAVE CSV ----------------
        df = pd.DataFrame(log_data,
                          columns=["Time", "Frame", "Mode", "Count", "Reason"])
        df.to_csv("crowd_log.csv", index=False)

        st.success("✅ Processing completed!")
        st.download_button("⬇ Download CSV Log", df.to_csv(index=False), "crowd_log.csv")