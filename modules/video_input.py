import cv2

class VideoInput:
    def __init__(self, video_path, skip_minutes=0):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"error: could not open video at {video_path}")
        if skip_minutes > 0:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, skip_minutes * 60 * 1000)
        
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()