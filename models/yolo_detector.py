import cv2
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize YOLOv8 detector.
        Args:
            model_path (str): Path to the YOLOv8 model weights. Defaults to 'yolov8n.pt'.
        """
        print(f"Loading YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)

    def detect(self, frame, conf_threshold=0.3):
        """
        Detect people in the frame using YOLOv8.
        Args:
            frame (numpy.ndarray): Input image frame.
            conf_threshold (float): Confidence threshold for detection.

        Returns:
            int: Number of people detected.
            list: List of bounding boxes [x1, y1, x2, y2].
            numpy.ndarray: Annotated frame.
        """
        results = self.model(frame, classes=[0], conf=conf_threshold, verbose=False) # class 0 is person
        
        count = 0
        boxes = []
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            count = len(boxes)
            annotated_frame = result.plot()
        
        return count, boxes, annotated_frame
