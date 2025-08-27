import cv2
import json
import math
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from ultralytics import YOLO


# ---------------- Utilities ----------------

def angle_3pts(a, b, c) -> float:
    """זווית ב-B בין נקודות a-b-c במעלות."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    ab = a - b
    cb = c - b
    nab = np.linalg.norm(ab) + 1e-6
    ncb = np.linalg.norm(cb) + 1e-6
    cosang = np.clip(np.dot(ab, cb) / (nab * ncb), -1.0, 1.0)
    return float(math.degrees(math.acos(cosang)))


def iou_xyxy(a, b) -> float:
    """IOU בין שתי תיבות בפורמט [x1,y1,x2,y2]."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    return float(inter / (area_a + area_b - inter + 1e-6))


# ---------------- Simple IOU Tracker ----------------

class SimpleTracker:
    """מעקב IDs בסיסי באמצעות שידוך IOU בין פריימים."""
    def __init__(self, iou_thresh=0.35, max_age=20):
        self.next_id = 1
        self.tracks = {}  # id -> {'bbox': [x1,y1,x2,y2], 'age': int}
        self.iou_thresh = iou_thresh
        self.max_age = max_age

    def update(self, boxes: List[List[float]]) -> List[int]:
        assigned_ids = [-1] * len(boxes)
        used = set()

        # נסה לשדך לכל track תיבה חדשה
        for tid, tinfo in list(self.tracks.items()):
            tbox = tinfo["bbox"]
            best_j, best_iou = -1, 0.0
            for j, box in enumerate(boxes):
                if j in used:
                    continue
                val = iou_xyxy(tbox, box)
                if val > best_iou:
                    best_iou, best_j = val, j
            if best_j != -1 and best_iou >= self.iou_thresh:
                assigned_ids[best_j] = tid
                self.tracks[tid] = {"bbox": boxes[best_j], "age": 0}
                used.add(best_j)
            else:
                tinfo["age"] += 1
                if tinfo["age"] > self.max_age:
                    del self.tracks[tid]

        # תיבות שלא שויכו → צור IDs חדשים
        for j, box in enumerate(boxes):
            if assigned_ids[j] == -1:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": box, "age": 0}
                assigned_ids[j] = tid

        return assigned_ids


# ---------------- Main Analyzer ----------------

class MotionAnalyzer:
    """
    מזהה base_state יחיד + events מרובים לכל אדם בפריים, ושומר JSON.
    """
    def __init__(self,
                 model_path: str = "yolov8n-pose.pt",
                 video_source: int | str = 0,
                 output_json: str = "motion_analysis_room1.json",
                 show: bool = True):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {video_source}")

        # מידע פריים
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        self.frame_size = (w, h)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

        # מעקבים
        self.tracker = SimpleTracker()
        self.prev_centers: Dict[int, Tuple[float, float]] = {}
        self.prev_areas: Dict[int, float] = {}

        # החלקה ל-base_state (רוב קולות על חלון קצר)
        self.base_hist: Dict[int, deque] = defaultdict(lambda: deque(maxlen=5))

        # פלט
        self.output_json = output_json
        self.frames_buffer: List[Dict[str, Any]] = []
        self.frame_id = 0

        self.show = show

    # --------- Feature extraction from pose ---------

    def _extract_person_features(self, kpts_xy: np.ndarray, bbox_xyxy: List[float]) -> Dict[str, Any]:
        """
        kpts_xy: [num_kpts, 2] קואורדינטות בפיקסלים
        bbox_xyxy: [x1,y1,x2,y2]
        """
        def safe_idx(i):
            if 0 <= i < len(kpts_xy):
                return kpts_xy[i]
            return np.array([np.nan, np.nan], dtype=float)

        # COCO keypoint indices (Ultralytics): 5/6 shoulders, 11/12 hips, 13/14 knees, 7/8 elbows, 9/10 wrists, 15/16 ankles
        r_sh, l_sh = safe_idx(5), safe_idx(6)
        r_hip, l_hip = safe_idx(11), safe_idx(12)
        r_knee, l_knee = safe_idx(13), safe_idx(14)
        r_elb, l_elb = safe_idx(7), safe_idx(8)
        r_wri, l_wri = safe_idx(9), safe_idx(10)

        # מרכז גוף: ממוצע כתפיים/ירכיים אם קיים, אחרת מרכז בוקס
        torso_pts = []
        for p in (r_sh, l_sh, r_hip, l_hip):
            if not np.any(np.isnan(p)):
                torso_pts.append(p)
        if torso_pts:
            center = np.mean(np.stack(torso_pts), axis=0)
        else:
            x1, y1, x2, y2 = bbox_xyxy
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)

        def safe_angle(a, b, c):
            if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
                return None
            return angle_3pts(a, b, c)

        knee_r = safe_angle(r_hip, r_knee, safe_idx(15))
        knee_l = safe_angle(l_hip, l_knee, safe_idx(16))
        hip_r = safe_angle(r_sh, r_hip, r_knee)
        hip_l = safe_angle(l_sh, l_hip, l_knee)

        # יד מורמת? פרק יד מעל הכתף (ב־OpenCV y קטן = גבוה יותר)
        right_hand_up = (not np.any(np.isnan(r_wri)) and not np.any(np.isnan(r_sh)) and r_wri[1] < r_sh[1])
        left_hand_up  = (not np.any(np.isnan(l_wri)) and not np.any(np.isnan(l_sh)) and l_wri[1] < l_sh[1])

        return {
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "knee_r": knee_r,
            "knee_l": knee_l,
            "hip_r": hip_r,
            "hip_l": hip_l,
            "right_hand_up": bool(right_hand_up),
            "left_hand_up": bool(left_hand_up),
            "right_wrist": None if np.any(np.isnan(r_wri)) else (float(r_wri[0]), float(r_wri[1])),
            "left_wrist":  None if np.any(np.isnan(l_wri)) else (float(l_wri[0]), float(l_wri[1])),
            "right_shoulder_y": None if np.any(np.isnan(r_sh)) else float(r_sh[1]),
            "left_shoulder_y":  None if np.any(np.isnan(l_sh)) else float(l_sh[1]),
        }

    # --------- Multi-label classifier (base_state + events) ---------

    def _classify_actions(self, pid: int, feats: Dict[str, Any], bbox: List[float]):
        """
        מחזיר:
          base_state: str (אחת)
          events: List[str] (0..N)
          conf: Dict[str,float] ציוני ביטחון
        """
        # מהירות (פיקסלים/פריים)
        prev_center = self.prev_centers.get(pid)
        cx, cy = feats["center_x"], feats["center_y"]
        speed = 0.0
        if prev_center is not None:
            dx = cx - prev_center[0]
            dy = cy - prev_center[1]
            speed = math.hypot(dx, dy)

        # ספים – כוונון לפי סצינה/רזולוציה
        FAST_T = 5.0
        SLOW_T = 1.0
        fast = speed > FAST_T
        slow = speed > SLOW_T

        # זוויות
        knees = [v for v in (feats.get("knee_r"), feats.get("knee_l")) if v is not None]
        hips  = [v for v in (feats.get("hip_r"),  feats.get("hip_l"))  if v is not None]
        mean_knee = float(np.mean(knees)) if knees else None
        mean_hip  = float(np.mean(hips))  if hips  else None

        events: List[str] = []
        conf: Dict[str, float] = {}

        # --- Event: raising_hand ---
        if feats.get("right_hand_up") or feats.get("left_hand_up"):
            events.append("raising_hand")
            conf["raising_hand"] = 0.9

        # --- Events: entering/exiting_room ---
        frame_w, frame_h = self.frame_size
        margin = 50  # פיקסלים מהקצה
        x1, y1, x2, y2 = bbox
        area = max(1.0, (x2 - x1) * (y2 - y1))
        prev_area = self.prev_areas.get(pid)
        area_ratio = (area / max(prev_area, 1.0)) if prev_area else None

        near_edge = (cx < margin) or (cx > frame_w - margin) or (cy > frame_h - margin)

        if near_edge and fast:
            entering_score = 0.0
            exiting_score = 0.0

            if prev_center is not None:
                prev_dist = math.hypot(prev_center[0] - frame_w / 2, prev_center[1] - frame_h / 2)
                curr_dist = math.hypot(cx - frame_w / 2, cy - frame_h / 2)
                moved_towards_center = curr_dist < prev_dist
                moved_away_center = curr_dist > prev_dist
            else:
                moved_towards_center = False
                moved_away_center = False

            getting_bigger  = (area_ratio is not None and area_ratio > 1.05)
            getting_smaller = (area_ratio is not None and area_ratio < 0.95)

            if moved_towards_center:
                entering_score += 0.6
            if getting_bigger:
                entering_score += 0.4

            if moved_away_center:
                exiting_score += 0.6
            if getting_smaller:
                exiting_score += 0.4

            if entering_score >= 0.7 and entering_score > exiting_score:
                events.append("entering_room")
                conf["entering_room"] = float(min(1.0, entering_score))
            elif exiting_score >= 0.7 and exiting_score > entering_score:
                events.append("exiting_room")
                conf["exiting_room"] = float(min(1.0, exiting_score))

        # --- Base state יחיד ---
        # היררכיה: ישיבה/כריעה > כיפוף > הליכה מהירה > תנועה קלה > עמידה
        base_state = "standing"
        base_conf = 0.65

        if mean_knee is not None and mean_hip is not None:
            if mean_knee < 110 and mean_hip < 140:
                base_state = "sitting_or_squatting"
                base_conf = 0.85
            elif mean_hip < 120:
                base_state = "bending"
                base_conf = 0.75
            else:
                if fast:
                    base_state = "walking_or_moving"
                    base_conf = 0.85
                elif slow:
                    base_state = "slight_movement"
                    base_conf = 0.7
                else:
                    base_state = "standing"
                    base_conf = 0.7
        else:
            # ללא זוויות תקינות → תבסס על תנועה
            if fast:
                base_state = "walking_or_moving"
                base_conf = 0.8
            elif slow:
                base_state = "slight_movement"
                base_conf = 0.65
            else:
                base_state = "standing"
                base_conf = 0.65

        conf["base_state"] = base_conf

        # עדכון להמשך
        self.prev_centers[pid] = (cx, cy)
        self.prev_areas[pid] = area

        return base_state, events, conf

    # --------- Optional smoothing for base_state ---------

    def _smooth_base_state(self, pid: int, state_now: str) -> str:
        """רוב קולות על חלון קצר כדי לייצב את מצב הבסיס."""
        hist = self.base_hist[pid]
        hist.append(state_now)
        # majority vote
        counts = defaultdict(int)
        for s in hist:
            counts[s] += 1
        return max(counts.items(), key=lambda kv: kv[1])[0]

    # --------- Main loop ---------

    def run(self):
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("End of stream.")
                    break

                results = self.model.predict(frame, verbose=False)
                res = results[0]

                # בנה רשימת תיבות
                boxes_xyxy: List[List[float]] = []
                person_indices: List[int] = []
                if res.boxes is not None and len(res.boxes) > 0:
                    for i in range(len(res.boxes)):
                        xyxy = res.boxes.xyxy[i].detach().cpu().numpy().tolist()
                        boxes_xyxy.append(xyxy)
                        person_indices.append(i)

                # עדכון מזהים
                ids = self.tracker.update(boxes_xyxy)

                # קייפוינטים
                kpts_xy = None
                if res.keypoints is not None and res.keypoints.xy is not None:
                    kpts_xy = res.keypoints.xy.detach().cpu().numpy()  # [N, K, 2]

                # רשומת פריים לפלט
                frame_entry = {
                    "frame_id": int(self.frame_id),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "detections": []
                }

                # עיבוד לכל אדם
                for idx, det_idx in enumerate(person_indices):
                    pid = int(ids[idx])
                    bbox = boxes_xyxy[idx]
                    kp = kpts_xy[idx] if kpts_xy is not None and idx < len(kpts_xy) else None

                    if kp is not None:
                        feats = self._extract_person_features(kp, bbox)
                    else:
                        # fallback ללא keypoints: מרכז = מרכז בוקס
                        x1, y1, x2, y2 = bbox
                        feats = {
                            "center_x": (x1 + x2) / 2.0,
                            "center_y": (y1 + y2) / 2.0,
                            "knee_r": None, "knee_l": None,
                            "hip_r": None,  "hip_l": None,
                            "right_hand_up": False, "left_hand_up": False,
                            "right_wrist": None, "left_wrist": None,
                            "right_shoulder_y": None, "left_shoulder_y": None
                        }

                    base_state_raw, events, conf = self._classify_actions(pid, feats, bbox)
                    base_state = self._smooth_base_state(pid, base_state_raw)

                    det = {
                        "id": pid,
                        "bbox": [float(x) for x in bbox],
                        "base_state": base_state,
                        "events": events,          # 0..N
                        "confidence": conf,        # כולל base_state
                        "features": feats
                    }
                    if kp is not None:
                        det["keypoints"] = kp.tolist()

                    frame_entry["detections"].append(det)

                # תצוגה
                if self.show:
                    annotated = res.plot()  # מצייר תיבות + שלדים
                    # כתוב טקסט לכל אדם
                    for det in frame_entry["detections"]:
                        x1, y1 = int(det["bbox"][0]), int(det["bbox"][1])
                        txt = f"id:{det['id']} {det['base_state']}"
                        if det["events"]:
                            txt += " | " + ",".join(det["events"])
                        cv2.putText(annotated, txt, (x1, max(12, y1 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow("Pose + BaseState + Events", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                self.frames_buffer.append(frame_entry)
                self.frame_id += 1

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self._write_output()

    def _write_output(self):
        out = {"frames": self.frames_buffer}
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Wrote {self.output_json} with {len(self.frames_buffer)} frames.")


if __name__ == "__main__":
    analyzer = MotionAnalyzer(
        model_path="yolov8n-pose.pt",   # דגם Pose (אפשר גם s/m לפי חומרה)
        video_source=0,                 # 0 = מצלמה; או path לקובץ וידאו
        output_json="motion_analysis_room1.json",
        show=True
    )
    analyzer.run()
