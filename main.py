import cv2
import numpy as np
import argparse
from models.yolo_detector import YoloDetector
from models.csrnet_estimator import CSRNetEstimator
from utils.alert_system import AlertSystem

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
    csv_writer.writerow(['Timestamp', 'Frame', 'Mode', 'Count', 'Alert'])

    # Initial state
    use_yolo = True
    frame_count = 0
    
    # For smoothing
    count_history = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Decision logic: Switch based on recent average count
        # To avoid flickering, we can use a hysteresis or just check the last count
        # For simplicity: if last count < threshold -> YOLO, else CSRNet
        # But if we are in CSRNet mode and count drops, we switch back.
        
        current_count = 0
        annotated_frame = frame.copy()
        
        if use_yolo:
            count, boxes, annotated_frame = yolo_model.detect(frame)
            current_count = count
            cv2.putText(annotated_frame, f"Mode: YOLO | Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Check if we should switch to CSRNet
            if count >= threshold:
                use_yolo = False
                print(f"Frame {frame_count}: Switching to CSRNet (High Density)")
        else:
            count, density_map = csrnet_model.estimate(frame)
            current_count = int(count)
            
            # Visualize density map
            # Normalize density map for display
            density_map_norm = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
            density_map_color = cv2.applyColorMap((density_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            density_map_color = cv2.resize(density_map_color, (width, height))
            
            # Overlay
            annotated_frame = cv2.addWeighted(frame, 0.6, density_map_color, 0.4, 0)
            cv2.putText(annotated_frame, f"Mode: CSRNet | Count: {int(count)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Check if we should switch to YOLO
            if count < threshold:
                use_yolo = True
                print(f"Frame {frame_count}: Switching to YOLO (Low Density)")

        # Alert Check
        triggered, msg = alert_system.check_alert(current_count)
        if triggered:
            cv2.putText(annotated_frame, "ALERT: HIGH DENSITY", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Log to CSV
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode_str = "YOLO" if use_yolo else "CSRNet"
        alert_str = "YES" if triggered else "NO"
        csv_writer.writerow([timestamp, frame_count, mode_str, current_count, alert_str])

        out.write(annotated_frame)
        
        # Optional: Display if running locally (not in typical Colab batch mode, though cv2_imshow helps there)
        # cv2.imshow('Crowd Control', annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

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
