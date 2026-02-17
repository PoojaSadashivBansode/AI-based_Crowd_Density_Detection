import cv2
import numpy as np
import argparse
from models.yolo_detector import YoloDetector
from models.csrnet_estimator import CSRNetEstimator
from utils.alert_system import AlertSystem

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
    return min(1.0, coverage)

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
        reasons.append(f"High count ({count} >= {count_threshold})")
    
    # Factor 2: Bounding box overlap
    overlap_ratio = calculate_box_overlap_ratio(boxes)
    if overlap_ratio >= 0.15:
        reasons.append(f"High overlap ({overlap_ratio:.2%})")
    
    # Factor 3: Crowd coverage
    coverage = calculate_crowd_coverage(boxes, frame_shape)
    if coverage >= 0.25:
        reasons.append(f"High coverage ({coverage:.2%})")
    
    # Factor 4: Sudden drop detection
    if len(prev_counts) >= 3:
        avg_recent = np.mean(prev_counts[-3:])
        if avg_recent > count_threshold and count < avg_recent * 0.7:
            reasons.append(f"Sudden drop ({count} < {avg_recent:.0f})")
    
    # Switch if at least 2 factors are triggered
    should_switch = len(reasons) >= 2
    reason_str = ", ".join(reasons) if reasons else "None"
    
    return should_switch, reason_str


def process_video(video_path, threshold=50, yolo_weights='yolov8n.pt', csrnet_weights='csrnet_weights.pth', output_path='output.mp4'):
    """
    Main processing loop for crowd density prediction.
    """
    # Initialize models
    print("Initializing models...")
    yolo_model = YoloDetector(yolo_weights)
    
    # Check if CSRNet weights exist, else warning
    try:
        csrnet_model = CSRNetEstimator(csrnet_weights)
    except FileNotFoundError:
        print(f"Warning: CSRNet weights not found at {csrnet_weights}. Running without custom weights (predictions will be inaccurate).")
        csrnet_model = CSRNetEstimator(None)
    except Exception as e:
        print(f"Error loading CSRNet: {e}")
        csrnet_model = CSRNetEstimator(None)

    alert_system = AlertSystem(threshold)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    # Video writer setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # CSV Logging Setup
    import csv
    import datetime
    log_file = open('crowd_data.csv', 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(['Timestamp', 'Frame', 'Mode', 'Count', 'Alert', 'Switch_Reason'])

    # Initial state
    frame_count = 0
    count_history = []  # Track count history for sudden drop detection
    
    print("Starting video processing with multi-factor model switching...")
    print("Factors: Count>=30, Overlap>=15%, Coverage>=25%, Sudden drops")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Always start with YOLO detection
        count, boxes, annotated_frame = yolo_model.detect(frame)
        mode = "YOLO"
        switch_reason = "Sparse crowd"
        
        # Multi-factor decision: Should we switch to CSRNet?
        should_switch, switch_reason = should_switch_to_csrnet(
            count, boxes, frame.shape, count_history, count_threshold=30
        )
        
        if should_switch:
            # Switch to CSRNet for dense crowd estimation
            c_count, density_map = csrnet_model.estimate(frame)
            calibration_factor = 0.18  # Fixed: was 20.0 (100x too high)
            count = int(abs(c_count) * calibration_factor)
            mode = "CSRNet"
            
            # Visualize density map
            density_map_norm = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
            density_map_color = cv2.applyColorMap((density_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            density_map_color = cv2.resize(density_map_color, (width, height))
            annotated_frame = cv2.addWeighted(frame, 0.6, density_map_color, 0.4, 0)
            
            if frame_count % 30 == 0:  # Log every 30 frames to avoid spam
                print(f"Frame {frame_count}: Using CSRNet - {switch_reason}")
        
        # Update count history (keep last 10)
        count_history.append(count)
        if len(count_history) > 10:
            count_history.pop(0)
        
        # Add model info overlay
        model_color = (0, 255, 255) if mode == "CSRNet" else (255, 150, 0)
        cv2.putText(annotated_frame, f"Model: {mode} | Count: {count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, model_color, 2)
        
        if switch_reason and switch_reason != "Sparse crowd":
            cv2.putText(annotated_frame, f"Reason: {switch_reason}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Alert Check
        triggered, msg = alert_system.check_alert(count)
        if triggered:
            cv2.putText(annotated_frame, "ALERT: HIGH DENSITY", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Log to CSV
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_str = "YES" if triggered else "NO"
        csv_writer.writerow([timestamp, frame_count, mode, count, alert_str, switch_reason])

        out.write(annotated_frame)

    cap.release()
    out.release()
    log_file.close()
    cv2.destroyAllWindows()
    print(f"Processing complete. Saved to {output_path}")
    print(f"Data logged to crowd_data.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='video.mp4', help='Path to input video')
    parser.add_argument('--threshold', type=int, default=50, help='Crowd density threshold')
    parser.add_argument('--yolo', type=str, default='yolov8s.pt', help='YOLO weights')
    parser.add_argument('--csrnet', type=str, default='csrnet_weights.pth', help='CSRNet weights')
    
    args = parser.parse_args()
    
    process_video(args.source, args.threshold, args.yolo, args.csrnet)
