# AI-based Crowd Density Prediction and Control System

This project implements a real-time crowd counting system using a hybrid approach:
- **YOLOv8** for low-to-medium density crowd detection.
- **CSRNet** for high-density crowd estimation.

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
├── app.py                 # Streamlit Dashboard
├── crowd_control.ipynb    # Google Colab notebook
├── download_weights.py    # Script to download CSRNet weights
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## How to Run the Dashboard (New!)

1.  **Install Streamlit**:
    Ensure you have all dependencies installed:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App**:
    Execute the following command in your terminal:
    ```bash
    python -m streamlit run app.py
    ```

3.  **Access**:
    The dashboard will automatically open in your default web browser (usually at `http://localhost:8501`).

## How to Run on Google Colab (with GPU)

1.  **Open the Notebook**:
    - Go to [Google Colab](https://colab.research.google.com/).
    - Click **File -> Upload notebook** and select the `crowd_control.ipynb` file from this repository.

2.  **Enable GPU**:
    - In Colab, go to **Runtime -> Change runtime type**.
    - Select **T4 GPU**.
    - Click **Save**.

3.  **Run the Cells**:
    - Run the cells sequentially.
    - **Note**: The notebook is now configured to **automatically download** the `csrnet_weights.pth` file. You do NOT need to upload it manually.

4.  **Upload Input Video**:
    - You still need to **manually upload** your `video.mp4` to the Colab session files (left sidebar).

5.  **Execute**:
    - Run the final cell to process the video.
    - Download `output.mp4` from the files sidebar.

## Local Usage (Script)

1.  **Download Weights**:
    Run the helper script to download CSRNet weights:
    ```bash
    python download_weights.py
    ```

2.  **Run Main Script**:
    ```bash
    python main.py --source video.mp4 --threshold 50
    ```
