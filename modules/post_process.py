import cv2

class PostProcessor:
    @staticmethod
    def annotate_tracks(frame, tracks):
        """
        Draws the tracked bounding boxes and unique IDs on the frame.
        """
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # Format: [left, top, right, bottom]
            x1, y1, x2, y2 = map(int, ltrb)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"player {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame
