import cv2
import numpy as np
import argparse
import os
from models.yolo_detector import YoloDetector
from models.csrnet_estimator import CSRNetEstimator
from utils.alert_system import AlertSystem

# ------------------ Helper Functions ------------------ #

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
    reason_str = ", ".join(reasons) if reasons else "Sparse crowd"
    
    return should_switch, reason_str


# ------------------ MAIN PROCESS FUNCTION ------------------ #

def process_input(source, threshold=50, yolo_weights='yolov8s.pt', csrnet_weights='csrnet_weights.pth', output_path='output.mp4'):

    print("Initializing models...")
    yolo_model = YoloDetector(yolo_weights)

    try:
        csrnet_model = CSRNetEstimator(csrnet_weights)
    except:
        print("CSRNet weights not found, using default init")
        csrnet_model = CSRNetEstimator(None)

    alert_system = AlertSystem(threshold)

    # Prepare output frame folder
    os.makedirs("output_frames", exist_ok=True)

    # Determine if input is video or frames folder
    is_frames = os.path.isdir(source)

    if is_frames:
        frame_files = sorted(os.listdir(source))
        cap = None
        print(f"Processing from frames folder: {source}")
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error opening video")
            return
        print(f"Processing video: {source}")

    # Setup writer only if video
    if not is_frames:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    else:
        out = None

    import csv, datetime
    log_file = open('crowd_data.csv', 'w', newline='')
    writer = csv.writer(log_file)
    writer.writerow(['Timestamp','Frame','Mode','Count','Alert','Reason'])

    frame_count = 0
    count_history = []

    while True:

        if is_frames:
            if frame_count >= len(frame_files):
                break
            frame_path = os.path.join(source, frame_files[frame_count])
            frame = cv2.imread(frame_path)
            if frame is None:
                frame_count += 1
                continue
        else:
            ret, frame = cap.read()
            if not ret:
                break

        frame_count += 1

        # YOLO detection
        count, boxes, annotated_frame = yolo_model.detect(frame)
        mode = "YOLO"

        # Decide CSRNet switch
        should_switch, reason = should_switch_to_csrnet(count, boxes, frame.shape, count_history)

        if should_switch:
            c_count, density_map = csrnet_model.estimate(frame)
            count = int(c_count * 0.20)
            mode = "CSRNet"

            density_map_norm = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
            density_map_color = cv2.applyColorMap((density_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            density_map_color = cv2.resize(density_map_color, (frame.shape[1], frame.shape[0]))
            annotated_frame = cv2.addWeighted(frame, 0.6, density_map_color, 0.4, 0)

        count_history.append(count)
        if len(count_history) > 10:
            count_history.pop(0)

        # Alert
        triggered, msg = alert_system.check_alert(count)

        # Overlay text
        cv2.putText(annotated_frame, f"Model: {mode} | Count: {count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

        if triggered:
            cv2.putText(annotated_frame,"ALERT!",(10,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

        # Save frame image
        cv2.imwrite(f"output_frames/frame_{frame_count:04d}.jpg", annotated_frame)

        # Save video
        if out is not None:
            out.write(annotated_frame)

        # CSV log
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        writer.writerow([timestamp, frame_count, mode, count, triggered, reason])

    # Release
    if cap:
        cap.release()
    if out:
        out.release()

    log_file.close()
    print("Processing complete ✅")
    print("Frames saved in folder: output_frames")
    print("CSV saved: crowd_data.csv")


# ------------------ MAIN ------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='video2.mp4')
    parser.add_argument('--threshold', type=int, default=50)
    parser.add_argument('--yolo', type=str, default='yolov8s.pt')
    parser.add_argument('--csrnet', type=str, default='csrnet_weights.pth')

    args = parser.parse_args()

    process_input(args.source, args.threshold, args.yolo, args.csrnet)