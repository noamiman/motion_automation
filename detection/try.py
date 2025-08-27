# smart_home_pose.py
# -*- coding: utf-8 -*-
"""
ניתוח תנוחות/תנועה לבית חכם (YOLO-Pose) עם ROI, נירמול מהירות, החלקה, וכתיבה אינקרמנטלית.
שומר NDJSON (שורה לכל פריים) + מטאדאטה, ויכול לייצר גם קובץ Summary בסיום.

תלויות: ultralytics, numpy, opencv-python, pyyaml (אופציונלי לקונפיג).
"""

from __future__ import annotations
import os, sys, json, math, time, pathlib, logging
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple, Deque
from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np
import cv2

try:
    import yaml  # אופציונלי
except Exception:
    yaml = None

from ultralytics import YOLO


# -------------------------- Utilities --------------------------

def angle_3pts(a, b, c) -> Optional[float]:
    """זווית ב-B בין נקודות a-b-c במעלות (או None אם חסר/NaN)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return None
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


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: str | pathlib.Path):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


# -------------------------- Config --------------------------

@dataclass
class ROIConfig:
    """ROI עבור כניסה/יציאה: אחד מאלה:
       - line: ((x1,y1),(x2,y2)) - חציית קו
       - box:  (x1,y1,x2,y2)     - כניסה/יציאה מאזור
    """
    line: Optional[List[List[float]]] = None
    box: Optional[List[float]] = None


@dataclass
class AnalyzerConfig:
    # קלט/מודל
    model_path: str = "yolov8n-pose.pt"
    video_source: str | int = 0  # 0=מצלמה
    camera_id: str = "room1_cam_a"
    room: str = "room1"

    # כתיבה/תיקיות
    runs_dir: str = "runs"
    run_name: Optional[str] = None  # אם None יווצר לפי תאריך
    write_every_n: int = 1          # כתיבת כל פריים (אפשר לרווח)
    ndjson_filename: str = "motion.ndjson"
    summary_filename: str = "summary.json"

    # תצוגה
    show: bool = True
    draw_roi: bool = True

    # מעקב/גדרות
    iou_thresh: float = 0.35
    max_age: int = 20

    # ספי מהירות (נירמול): חלק יחסי מהאלכסון לשנייה
    fast_norm: float = 0.12
    slow_norm: float = 0.02

    # החלקה
    base_state_hist: int = 5
    ema_speed_alpha: float = 0.7  # 0..1

    # זיהוי יד מורמת (יחסי לטורסו)
    hand_up_torso_ratio: float = 0.15

    # ROI (לא חובה)
    roi: ROIConfig = field(default_factory=ROIConfig)

    # ספים נוספים
    keypoint_min_conf: float = 0.5  # אם קיים res.keypoints.conf

    # לוג
    log_level: str = "INFO"


# -------------------------- Simple IOU Tracker --------------------------

class SimpleTracker:
    """מעקב IDs בסיסי באמצעות IOU בין פריימים."""
    def __init__(self, iou_thresh=0.35, max_age=20):
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}  # id -> {'bbox': xyxy, 'age': int}
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
                tinfo["age"] = tinfo.get("age", 0) + 1
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


# -------------------------- Analyzer --------------------------

class SmartHomePoseAnalyzer:
    def __init__(self, cfg: AnalyzerConfig):
        self.cfg = cfg

        # לוג
        logging.basicConfig(
            level=getattr(logging, cfg.log_level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)s | %(message)s"
        )
        self.log = logging.getLogger("SmartHomePose")

        # תיקיית ריצה
        run_name = cfg.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(cfg.runs_dir, run_name)
        ensure_dir(self.run_dir)

        # קבצים
        self.ndjson_path = os.path.join(self.run_dir, cfg.ndjson_filename)
        self.summary_path = os.path.join(self.run_dir, cfg.summary_filename)

        # מודל + וידאו
        self.model = YOLO(cfg.model_path)
        self.cap = cv2.VideoCapture(cfg.video_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {cfg.video_source}")

        # מאפייני פריים
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        self.frame_size = (w, h)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0  # ברירת מחדל 30
        self.diag = math.hypot(*self.frame_size) + 1e-6

        # מעקבים וזיכרונות
        self.tracker = SimpleTracker(iou_thresh=cfg.iou_thresh, max_age=cfg.max_age)
        self.prev_centers: Dict[int, Tuple[float, float]] = {}
        self.prev_areas: Dict[int, float] = {}
        self.speed_ema: Dict[int, float] = {}

        self.base_hist: Dict[int, Deque[str]] = defaultdict(lambda: deque(maxlen=cfg.base_state_hist))

        # מדדים
        self.frame_id = 0
        self.frames_with_person = 0
        self.total_tracks_seen = 0
        self.track_lifetimes: Dict[int, int] = defaultdict(int)

        # הכנה לקובץ NDJSON
        with open(self.ndjson_path, "w", encoding="utf-8") as f:
            meta = {
                "schema_version": "1.0",
                "type": "meta",
                "camera_id": cfg.camera_id,
                "room": cfg.room,
                "model_path": cfg.model_path,
                "video_source": str(cfg.video_source),
                "fps": self.fps,
                "frame_size": {"w": w, "h": h},
                "created_utc": now_utc_iso(),
                "config": asdict(cfg)
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        self.log.info(f"Running to {self.run_dir} | fps~{self.fps:.1f} | size={self.frame_size}")

    # -------------------- Feature extraction --------------------

    def _extract_features(self, kpts_xy: np.ndarray, kpts_conf: Optional[np.ndarray], bbox_xyxy: List[float]) -> Dict[str, Any]:
        """חילוץ פיצ'רים מאדם אחד."""
        def safe_idx(i):
            if 0 <= i < len(kpts_xy):
                return kpts_xy[i]
            return np.array([np.nan, np.nan], dtype=float)

        def conf_ok(i):
            if kpts_conf is None:
                return True
            if 0 <= i < len(kpts_conf):
                return float(kpts_conf[i]) >= self.cfg.keypoint_min_conf
            return False

        # אינדקסים לפי COCO (Ultralytics)
        # 0 nose, 1 L-eye, 2 R-eye, 3 L-ear, 4 R-ear,
        # 5 L-shoulder, 6 R-shoulder, 7 L-elbow, 8 R-elbow,
        # 9 L-wrist, 10 R-wrist, 11 L-hip, 12 R-hip,
        # 13 L-knee, 14 R-knee, 15 L-ankle, 16 R-ankle
        L_SH, R_SH = 5, 6
        L_HIP, R_HIP = 11, 12
        L_KNEE, R_KNEE = 13, 14
        L_ANK, R_ANK = 15, 16
        L_WRI, R_WRI = 9, 10

        r_sh, l_sh = safe_idx(R_SH), safe_idx(L_SH)
        r_hip, l_hip = safe_idx(R_HIP), safe_idx(L_HIP)
        r_knee, l_knee = safe_idx(R_KNEE), safe_idx(L_KNEE)
        r_wri, l_wri = safe_idx(R_WRI), safe_idx(L_WRI)

        # מרכז גוף: ממוצע כתפיים/ירכיים (אם מוכר) אחרת מרכז בוקס
        torso_pts = []
        for i, p in [(R_SH, r_sh), (L_SH, l_sh), (R_HIP, r_hip), (L_HIP, l_hip)]:
            if conf_ok(i) and not np.any(np.isnan(p)):
                torso_pts.append(p)
        if torso_pts:
            center = np.mean(np.stack(torso_pts), axis=0)
        else:
            x1, y1, x2, y2 = bbox_xyxy
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)

        # זוויות (רק אם כל הנקודות עם מספיק קונפידנס)
        knee_r = angle_3pts(r_hip if conf_ok(R_HIP) else [np.nan, np.nan],
                            r_knee if conf_ok(R_KNEE) else [np.nan, np.nan],
                            safe_idx(R_ANK) if conf_ok(R_ANK) else [np.nan, np.nan])
        knee_l = angle_3pts(l_hip if conf_ok(L_HIP) else [np.nan, np.nan],
                            l_knee if conf_ok(L_KNEE) else [np.nan, np.nan],
                            safe_idx(L_ANK) if conf_ok(L_ANK) else [np.nan, np.nan])

        hip_r = angle_3pts(r_sh if conf_ok(R_SH) else [np.nan, np.nan],
                           r_hip if conf_ok(R_HIP) else [np.nan, np.nan],
                           r_knee if conf_ok(R_KNEE) else [np.nan, np.nan])
        hip_l = angle_3pts(l_sh if conf_ok(L_SH) else [np.nan, np.nan],
                           l_hip if conf_ok(L_HIP) else [np.nan, np.nan],
                           l_knee if conf_ok(L_KNEE) else [np.nan, np.nan])

        # יד מורמת ביחס לגובה טורסו
        if all(conf_ok(i) for i in (L_SH, R_SH, L_HIP, R_HIP)):
            shoulder_y_mean = np.nanmean([l_sh[1], r_sh[1]])
            hip_y_mean = np.nanmean([l_hip[1], r_hip[1]])
            torso_h = abs(shoulder_y_mean - hip_y_mean) + 1e-6
            tol = self.cfg.hand_up_torso_ratio * torso_h
        else:
            shoulder_y_mean = None
            tol = 0.0

        def hand_up(wrist, shoulder, wrist_ok, shoulder_ok):
            if not wrist_ok or not shoulder_ok or np.any(np.isnan(wrist)) or np.any(np.isnan(shoulder)):
                return False
            # y קטן = גבוה יותר
            return wrist[1] < (shoulder[1] - tol)

        right_hand_up = hand_up(r_wri, r_sh, conf_ok(R_WRI), conf_ok(R_SH))
        left_hand_up = hand_up(l_wri, l_sh, conf_ok(L_WRI), conf_ok(L_SH))

        feats = {
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "knee_r": knee_r,
            "knee_l": knee_l,
            "hip_r": hip_r,
            "hip_l": hip_l,
            "right_hand_up": bool(right_hand_up),
            "left_hand_up": bool(left_hand_up),
        }

        # נשמור wrist/shoulder רק אם קיימים (לא חובה)
        feats["right_wrist"] = None if not conf_ok(R_WRI) or np.any(np.isnan(r_wri)) else (float(r_wri[0]), float(r_wri[1]))
        feats["left_wrist"]  = None if not conf_ok(L_WRI) or np.any(np.isnan(l_wri)) else (float(l_wri[0]), float(l_wri[1]))

        return feats

    # -------------------- Action classification --------------------

    def _classify(self, pid: int, feats: Dict[str, Any], bbox: List[float]) -> Tuple[str, List[str], Dict[str, float]]:
        """קביעת מצב בסיס + אירועים. מחזיר (base_state, events, conf_dict)."""
        # מהירות (פיקסלים/פריים → יחס אלכסון לשנייה)
        prev_center = self.prev_centers.get(pid)
        cx, cy = feats["center_x"], feats["center_y"]
        speed_ppf = 0.0
        if prev_center is not None:
            dx = cx - prev_center[0]
            dy = cy - prev_center[1]
            speed_ppf = math.hypot(dx, dy)
        px_per_sec = speed_ppf * self.fps
        norm_speed = px_per_sec / self.diag  # 0..1 בקירוב

        # EMA למהירות
        ema = self.speed_ema.get(pid, norm_speed)
        alpha = self.cfg.ema_speed_alpha
        ema = alpha * ema + (1 - alpha) * norm_speed
        self.speed_ema[pid] = ema

        fast = ema > self.cfg.fast_norm
        slow = ema > self.cfg.slow_norm

        # זוויות
        knees = [v for v in (feats.get("knee_r"), feats.get("knee_l")) if v is not None]
        hips  = [v for v in (feats.get("hip_r"),  feats.get("hip_l"))  if v is not None]
        mean_knee = float(np.mean(knees)) if knees else None
        mean_hip  = float(np.mean(hips))  if hips  else None

        events: List[str] = []
        conf: Dict[str, float] = {}

        # יד מורמת
        if feats.get("right_hand_up") or feats.get("left_hand_up"):
            events.append("raising_hand")
            conf["raising_hand"] = 0.9

        # כניסה/יציאה לפי ROI
        x1, y1, x2, y2 = bbox
        area = max(1.0, (x2 - x1) * (y2 - y1))
        prev_area = self.prev_areas.get(pid)
        area_ratio = (area / max(prev_area, 1.0)) if prev_area else None

        def line_side(p1, p2, p):
            return np.sign(np.cross(p2 - p1, p - p1))

        if self.cfg.roi.line is not None and len(self.cfg.roi.line) == 2:
            p1 = np.array(self.cfg.roi.line[0], dtype=float)
            p2 = np.array(self.cfg.roi.line[1], dtype=float)
            crossed = False
            moving_to_center = False
            if prev_center is not None:
                prev_side = line_side(p1, p2, np.array(prev_center))
                curr_side = line_side(p1, p2, np.array([cx, cy]))
                crossed = (prev_side != 0 and curr_side != 0 and prev_side != curr_side)
                # כיוון כללי: מרחק מהמרכז (מסייע להכריע Enter/Exit)
                frame_w, frame_h = self.frame_size
                prev_dist = math.hypot(prev_center[0] - frame_w / 2, prev_center[1] - frame_h / 2)
                curr_dist = math.hypot(cx - frame_w / 2, cy - frame_h / 2)
                moving_to_center = curr_dist < prev_dist

            if crossed and fast:
                if moving_to_center or (area_ratio is not None and area_ratio > 1.05):
                    events.append("entering_room")
                    conf["entering_room"] = 0.85
                elif (not moving_to_center) or (area_ratio is not None and area_ratio < 0.95):
                    events.append("exiting_room")
                    conf["exiting_room"] = 0.85

        elif self.cfg.roi.box is not None and len(self.cfg.roi.box) == 4:
            bx1, by1, bx2, by2 = self.cfg.roi.box
            inside_prev = False
            inside_now = False
            if prev_center is not None:
                inside_prev = (bx1 <= prev_center[0] <= bx2) and (by1 <= prev_center[1] <= by2)
            inside_now = (bx1 <= cx <= bx2) and (by1 <= cy <= by2)
            if inside_prev != inside_now and fast:
                if inside_now:
                    events.append("entering_room")
                    conf["entering_room"] = 0.8
                else:
                    events.append("exiting_room")
                    conf["exiting_room"] = 0.8

        # מצב בסיס (יחיד)
        base_state = "standing"
        base_conf = 0.65
        if mean_knee is not None and mean_hip is not None:
            if mean_knee < 110 and mean_hip < 140:
                base_state, base_conf = "sitting_or_squatting", 0.85
            elif mean_hip < 120:
                base_state, base_conf = "bending", 0.75
            else:
                if fast:
                    base_state, base_conf = "walking_or_moving", 0.85
                elif slow:
                    base_state, base_conf = "slight_movement", 0.70
                else:
                    base_state, base_conf = "standing", 0.70
        else:
            if fast:
                base_state, base_conf = "walking_or_moving", 0.80
            elif slow:
                base_state, base_conf = "slight_movement", 0.65
            else:
                base_state, base_conf = "standing", 0.65

        conf["base_state"] = base_conf

        # עדכונים להמשך
        self.prev_centers[pid] = (cx, cy)
        self.prev_areas[pid] = area

        return base_state, events, conf

    def _smooth_base(self, pid: int, state_now: str) -> str:
        hist = self.base_hist[pid]
        hist.append(state_now)
        counts = defaultdict(int)
        for s in hist:
            counts[s] += 1
        return max(counts.items(), key=lambda kv: kv[1])[0]

    # -------------------- Drawing --------------------

    def _draw(self, frame, res, detections: List[Dict[str, Any]]):
        annotated = res.plot()
        # ROI
        if self.cfg.draw_roi:
            if self.cfg.roi.line:
                (x1, y1), (x2, y2) = self.cfg.roi.line
                cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            if self.cfg.roi.box:
                x1, y1, x2, y2 = map(int, self.cfg.roi.box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)

        for det in detections:
            bbox = det.get("bbox_xyxy")
            if not bbox or len(bbox) < 2:
                continue
            x1, y1 = int(bbox[0]), int(bbox[1])
            txt = f"id:{det.get('id', '?')} {det.get('base_state', '')}"
            ev = det.get("events") or []
            if ev:
                txt += " | " + ",".join(ev)
            cv2.putText(annotated, txt, (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        return annotated

    # -------------------- IO --------------------

    def _write_frame_ndjson(self, record: Dict[str, Any]):
        with open(self.ndjson_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_summary(self):
        # מדדים
        avg_track_len = 0.0
        track_count = len(self.track_lifetimes)
        if track_count:
            avg_track_len = float(np.mean(list(self.track_lifetimes.values())))
        summary = {
            "schema_version": "1.0",
            "type": "summary",
            "camera_id": self.cfg.camera_id,
            "room": self.cfg.room,
            "frames_total": self.frame_id,
            "frames_with_person": self.frames_with_person,
            "frames_with_person_ratio": (self.frames_with_person / max(1, self.frame_id)),
            "tracks_total": self.total_tracks_seen,
            "avg_track_length_frames": avg_track_len,
            "finished_utc": now_utc_iso()
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        self.log.info(f"Summary written: {self.summary_path}")

    # -------------------- Main loop --------------------

    def run(self):
        t0 = time.time()
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    self.log.info("End of stream.")
                    break

                results = self.model.predict(frame, verbose=False)
                res = results[0]

                # בנה רשימת תיבות (person בלבד בדגמי pose; עדיין נשמור כלליות)
                boxes_xyxy: List[List[float]] = []
                if res.boxes is not None and len(res.boxes) > 0:
                    boxes_xyxy = res.boxes.xyxy.detach().cpu().numpy().tolist()

                # עדכון מזהים
                ids = self.tracker.update(boxes_xyxy)

                # קייפוינטים
                kpts_xy = None
                kpts_conf = None
                if res.keypoints is not None:
                    if res.keypoints.xy is not None:
                        kpts_xy = res.keypoints.xy.detach().cpu().numpy()  # [N, K, 2]
                    if hasattr(res.keypoints, "conf") and res.keypoints.conf is not None:
                        # חלק מהגרסאות מחזירות [N,K], לעיתים None
                        try:
                            kpts_conf = res.keypoints.conf.detach().cpu().numpy()
                        except Exception:
                            kpts_conf = None

                # רשומת פריים
                frame_record: Dict[str, Any] = {
                    "type": "frame",
                    "frame_id": int(self.frame_id),
                    "ts_utc": now_utc_iso(),
                    "camera_id": self.cfg.camera_id,
                    "room": self.cfg.room,
                    "fps": self.fps,
                    "frame_size": {"w": self.frame_size[0], "h": self.frame_size[1]},
                    "detections": []
                }

                # עיבוד לכל דיטקציה
                persons_here = 0
                for i, bbox in enumerate(boxes_xyxy):
                    pid = int(ids[i])
                    kp_xy = kpts_xy[i] if (kpts_xy is not None and i < len(kpts_xy)) else None
                    kp_conf = kpts_conf[i] if (kpts_conf is not None and i < len(kpts_conf)) else None

                    if kp_xy is not None:
                        feats = self._extract_features(kp_xy, kp_conf, bbox)
                    else:
                        # fallback: מרכז = מרכז bbox
                        x1, y1, x2, y2 = bbox
                        feats = {
                            "center_x": (x1 + x2) / 2.0,
                            "center_y": (y1 + y2) / 2.0,
                            "knee_r": None, "knee_l": None,
                            "hip_r": None, "hip_l": None,
                            "right_hand_up": False, "left_hand_up": False,
                            "right_wrist": None, "left_wrist": None,
                        }

                    base_state_raw, events, conf = self._classify(pid, feats, bbox)
                    base_state = self._smooth_base(pid, base_state_raw)

                    det = {
                        "id": pid,
                        "bbox_xyxy": [float(x) for x in bbox],
                        "base_state": base_state,
                        "events": events,
                        "confidence": conf,
                        "features": feats
                    }
                    # אופציונלי: שמירת keypoints עצמם (עלול להיות כבד)
                    # אם תרצה לשמור: בטל את ההערה:
                    # if kp_xy is not None:
                    #     det["keypoints_xy"] = kp_xy.tolist()
                    #     if kp_conf is not None:
                    #         det["keypoints_conf"] = kp_conf.tolist()

                    frame_record["detections"].append(det)
                    persons_here += 1

                    # סטטיסטיקות track
                    self.track_lifetimes[pid] += 1
                    self.total_tracks_seen = max(self.total_tracks_seen, pid)

                if persons_here > 0:
                    self.frames_with_person += 1

                # כתיבה אינקרמנטלית
                if (self.frame_id % self.cfg.write_every_n) == 0:
                    self._write_frame_ndjson(frame_record)

                # תצוגה
                if self.cfg.show:
                    annotated = self._draw(frame, res, frame_record["detections"])
                    cv2.imshow("SmartHome Pose (YOLO)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                self.frame_id += 1

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self._write_summary()
            t1 = time.time()
            self.log.info(f"Done. frames={self.frame_id} | time={t1 - t0:.1f}s")

# -------------------------- Helpers --------------------------

def load_config(path: Optional[str]) -> AnalyzerConfig:
    """טען קונפיג מ-YAML אם path ניתן וקיים; אחרת החזר דיפולט."""
    cfg = AnalyzerConfig()
    if path and os.path.isfile(path):
        if yaml is None:
            raise RuntimeError("PyYAML אינו מותקן אך סופק קובץ YAML. התקן pyyaml או הסר את הקובץ.")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        # תמיכה ב-ROI
        roi_raw = raw.pop("roi", None)
        cfg = AnalyzerConfig(**raw)
        if roi_raw:
            cfg.roi = ROIConfig(**roi_raw)
    return cfg

# -------------------------- Main --------------------------

if __name__ == "__main__":
    """
    דוגמה לקובץ config.yaml (אופציונלי):
    ---
    model_path: yolov8n-pose.pt
    video_source: 0
    camera_id: room1_cam_a
    room: room1
    runs_dir: runs
    run_name: null          # או "room1_test"
    write_every_n: 1
    show: true
    draw_roi: true
    iou_thresh: 0.35
    max_age: 20
    fast_norm: 0.12
    slow_norm: 0.02
    base_state_hist: 5
    ema_speed_alpha: 0.7
    hand_up_torso_ratio: 0.15
    keypoint_min_conf: 0.5
    log_level: INFO
    roi:
      line: [[50, 540], [1830, 540]]  # קו מעבר אופקי לדוגמה
      # box: [x1, y1, x2, y2]
    """
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = load_config(cfg_path)

    analyzer = SmartHomePoseAnalyzer(cfg)
    analyzer.run()
