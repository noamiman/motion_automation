import cv2
import json
from datetime import datetime, timezone
from ultralytics import YOLO
from collections import Counter
import os

class BasicObjectDetection:
    def __init__(self, model_path, basic_home_path,
                 output_json="detections.json",
                 room_name="room1",
                 confidence_threshold=0.6,
                 video_source=0
                 ):
        """
        :param model_path: נתיב למודל YOLO
        :param basic_home_path: קובץ JSON עם מיפוי אובייקטים לפי חדר
        :param output_json: קובץ פלט לכתיבת הזיהויים
        :param room_name: שם החדר לניתוח
        :param confidence_threshold: סף אמון מינימלי
        :param video_source: מקור וידאו - 0 (מצלמה), או path לסרטון
        """
        self.model = YOLO(model_path)

        with open(basic_home_path, "r") as f:
            self.basic_home_details = json.load(f)

        self.output_json = output_json
        self.room_name = room_name
        self.confidence_threshold = confidence_threshold
        self.video_source = video_source

        self.cap = cv2.VideoCapture(video_source)
        self.all_frames_data = []
        self.frame_id = 0

        # Reset output file
        if not os.path.exists(self.output_json):
            with open(self.output_json, "w") as f:
                json.dump([], f)


    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of stream or cannot read frame.")
                break

            results = self.model(frame, verbose=False)[0]
            object_counter = Counter()

            frame_data = {
                "frame_id": self.frame_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "detections": [],
                "object_counter": None
            }

            room_objects = self.basic_home_details[self.room_name]["objects"]
            id_to_name = {v: k for k, v in room_objects.items()}

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id in id_to_name and conf >= self.confidence_threshold:
                    label_name = id_to_name[cls_id]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox = [x1, y1, x2 - x1, y2 - y1]

                    object_counter[label_name] += 1

                    frame_data["detections"].append({
                        "label": label_name,
                        "confidence": conf,
                        "bbox": bbox
                    })

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label_name} {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if object_counter:
                frame_data["object_counter"] = dict(object_counter)
                # print(f"Frame {self.frame_id}:", object_counter)

            if frame_data["detections"]:
                self.all_frames_data.append(frame_data)

            self.frame_id += 1

            cv2.imshow("YOLOv8 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self._end()

    def get_all_frames_data(self):
        """
        מחזירה את כל המידע לפי פריימים:
        - frame_id
        - timestamp
        - detections (רשימה של תווית, קונפידנס, bbox)
        - object_counter (ספירה לפי תווית)
        """
        return self.all_frames_data

    def get_frame_by_id(self, frame_id):
        for frame in self.all_frames_data:
            if int(frame["frame_id"]) == frame_id:
                return frame
        return None  # אם לא נמצא שום פריים עם ה-ID הזה

    def get_detections_by_label(self, labels):
        if not isinstance(labels, list):
            labels = [labels]

        detections = []
        for frame in self.all_frames_data:
            for det in frame["detections"]:
                if det["label"] in labels:
                    det_with_frame = det.copy()
                    det_with_frame["frame_id"] = frame["frame_id"]
                    detections.append(det_with_frame)
        return detections
    def get_counter_by_id(self, frame_id):
            for frame in self.all_frames_data:
                if int(frame["frame_id"]) == frame_id:
                    return frame["object_counter"]
            return None  # אם לא נמצא שום פריים עם ה-ID הזה

    def show_detections(self):
        frames = self.get_all_frames_data()
        for frame in frames:
            print(f"Frame {frame['frame_id']} @ {frame['timestamp']}")
            print("Detections:", frame["detections"])
            print("Counts:", frame["object_counter"])
            print("-" * 30)

    def _end(self):
        self.cap.release()
        cv2.destroyAllWindows()

        with open(self.output_json, "w") as f:
            json.dump(self.all_frames_data, f, indent=2)

        print(f"\nDetection data saved to {self.output_json}")

detector = BasicObjectDetection(
    model_path="yolov8n.pt",
    basic_home_path="data/basicHome.json",
    output_json="detections_room1.json",
    room_name="room1",
    confidence_threshold=0.6,
    video_source=0       # 0 = מצלמה חיה
)

detector.run()  # מפעיל את הזיהוי ושומר את הנתונים

print(detector.get_counter_by_id(5))