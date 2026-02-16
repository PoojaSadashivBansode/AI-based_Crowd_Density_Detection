# AI-based Crowd Density Prediction and Control System

This project implements a real-time crowd counting system using a hybrid approach:
- **YOLOv8** for low-to-medium density crowd detection (bounding box based).
- **CSRNet** for high-density crowd estimation (density map based).

The system automatically switches between models based on a configurable density threshold and triggers alerts when the count exceeds safe limits.

## Project Structure
```
Crowd_detection/
├── models/
│   ├── yolo_detector.py   # YOLOv8 wrapper
│   └── csrnet_estimator.py # CSRNet wrapper and model definition
├── utils/
│   ├── alert_system.py    # Logic for triggering alerts
│   └── video_stream.py    # Video capture utility
├── main.py                # Main execution script
├── crowd_control.ipynb    # Google Colab notebook
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Setup & Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Weights**:
   - **YOLOv8**: `yolov8n.pt` will be automatically downloaded by the `ultralytics` library on first run.
   - **CSRNet**: You need to place the pre-trained CSRNet weights file (e.g., `csrnet_weights.pth`) in the project root. If not provided, the system will run with initialized (random) weights for the CSRNet part (warning will be displayed).

## Usage

### Local Execution
Run the `main.py` script:
```bash
python main.py --source video.mp4 --threshold 50
```

**Arguments**:
- `--source`: Path to video file or `0` for webcam.
- `--threshold`: Count threshold for switching to CSRNet and triggering high-density alerts.
- `--yolo`: Path to YOLO weights (default: `yolov8n.pt`).
- `--csrnet`: Path to CSRNet weights (default: `csrnet_weights.pth`).

### Google Colab Execution
1. Upload the `Crowd_detection` folder to your Google Drive.
2. Open `crowd_control.ipynb` in Google Colab.
3. specific paths in the notebook to match your Drive structure (e.g., `/content/drive/MyDrive/BE Project/Crowd_detection`).
4. Run the cells to install dependencies and execute the crowd analysis.

## Alert System
- **Console Alert**: Prints `ALERT: HIGH DENSITY` in red text when count > threshold.
- **Visual Alert**: Overlays "ALERT: HIGH DENSITY" on the output video.
