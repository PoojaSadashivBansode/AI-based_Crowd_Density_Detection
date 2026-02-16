import cv2

class VideoStream:
    def __init__(self, source=0):
        """
        Initialize Video Stream.
        Args:
            source (int or str): Video source. 0 for webcam, or path to video file.
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
    def get_frame(self):
        """
        Yields frames from the video source.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
            
    def release(self):
        """
        Release the video source.
        """
        self.cap.release()
