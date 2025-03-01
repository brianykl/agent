import cv2

class Display:
    def __init__(self, window_name="YOLOv8 - player detection"):
        self.window_name = window_name

    def show_frame(self, frame):
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key

    def close(self):
        cv2.destroyAllWindows()
