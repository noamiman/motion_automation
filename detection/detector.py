import cv2
import json
from datetime import datetime, timezone
from torch.distributed.tensor import empty
from ultralytics import YOLO
from collections import Counter

# using yolov8 model for object detection
model_path = "yolov8n.pt"
model = YOLO(model_path)
# print the model lables and ID's
print(model.names)

# using rooms objects that already defined
with open("basicHome.json", "r") as f:
    basic_home_details = json.load(f)

# open camera
cap = cv2.VideoCapture(0)


# reset json
with open("detections.json", "w") as f:
    json.dump([], f)

# reset parameters
all_frames_data = []
frame_id = 0
room_name = "room1"

# general loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # using the model for detection
    results = model(frame, verbose=False)[0]
    frame_data = {
        "frame_id": frame_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detections": []
    }


    room_objects = basic_home_details[room_name]["objects"]
    id_to_name = {v: k for k, v in room_objects.items()}
    object_counter = Counter()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in id_to_name:
            label_name = id_to_name[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [x1, y1, x2 - x1, y2 - y1]
            object_counter[label_name] += 1

            if conf<0.6:
                continue

            frame_data["detections"].append({
                "label": label_name,
                "confidence": conf,
                "bbox": bbox
            })

            # ציור תיבות גם כן בתוך התנאי — אבל ישאיר את התיבה בפריים הנוכחי
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    all_frames_data.append(frame_data)

    frame_id += 1  # הגדלה תמידית של frame_id
    cv2.imshow("YOLOv8 Detection", frame)
    # יציאה מהלולאה
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



# כשהלולאה נגמרת:
cap.release()
cv2.destroyAllWindows()

# שמירה ל-JSON
with open("detections.json", "w") as f:
    json.dump(all_frames_data, f, indent=2)



# פעולה שמקבלת 2 מילונים ובודקת את ההבדלים ביניהם, מילון של לפני ומילון של אחרי ואז נשווה
#לדוגמה person: 2
# person: 1

def compare_counters(before, after):
    differences = {}
    all_keys = set(before.keys()).union(after.keys())

    for key in all_keys:
        before_count = before.get(key, 0)
        after_count = after.get(key, 0)
        delta = after_count - before_count

        if delta != 0:
            direction = "increased" if delta > 0 else "decreased"
            differences[key] = {
                "change": direction,
                "amount": abs(delta),
                "from": before_count,
                "to": after_count
            }

    if not differences:
        return "No Change"
    return differences

