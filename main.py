from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

model = YOLO("yolov8n.pt")

video_path = "videos/game0.mp4"
cap = cv2.VideoCapture(video_path)

tracker = DeepSort(max_age=30)

def dumbo():
    frame_count = 0
    max_frames = 30 * 120 

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])

                if model.names[cls] == 'person' and conf > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "player", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id  
            ltrb = track.to_ltrb()     
            x1, y1, x2, y2 = map(int, ltrb)

            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"player {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 - player detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("Hello from agent!")
    dumbo()


if __name__ == "__main__":
    main()
