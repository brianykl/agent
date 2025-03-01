from modules.video_input import VideoInput
from modules.detector import Detector
from modules.tracker import Tracker
from modules.post_process import PostProcessor
from modules.display import Display

def main():
    video_path = "videos/game0.mp4"
    
    video_input = VideoInput(video_path, 3)
    detector = Detector(model_path="yolov8n.pt", conf_threshold=0.5)
    tracker = Tracker(max_age=30)
    display = Display(window_name="YOLOv8 - player detection")
    
    frame_count = 30 * 30
    max_frames = 30 * 120  
    
    while frame_count < max_frames:
        frame = video_input.get_frame()
        if frame is None:
            break
        
        detections = detector.detect(frame)
        
        frame = detector.annotate(frame)
        
        tracks = tracker.update(detections, frame)
        
        # frame = PostProcessor.annotate_tracks(frame, tracks)
        
        key = display.show_frame(frame)
        if key == ord('q'):
            break
        
        frame_count += 1

    video_input.release()
    display.close()

if __name__ == "__main__":
    main()
