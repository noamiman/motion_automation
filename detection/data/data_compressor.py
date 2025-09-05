import json
from idlelib.iomenu import encoding
from typing import Any, Dict, List, Union

from ultralytics.hub import events


# מיפוי ערכי זווית -> קטגוריה מילולית
def _joint_cat(val: Any) -> str:
    if val is None:
        return "unknown"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "unknown"
    # ספים דיפולטיביים סבירים: <10 ישר, 10–40 כפיפה קלה, >40 כפיפה עמוקה
    if v < 10:
        return "straight"
    if v < 40:
        return "slight_bend"
    return "deep_bend"

def _conf_bin(p: float) -> str:
    # high ≥ 0.8, medium ≥ 0.5, אחרת low
    if p >= 0.8:
        return "high"
    if p >= 0.5:
        return "medium"
    return "low"

def _reach_from_hands(left_up: bool, right_up: bool) -> str:
    if right_up and not left_up:
        return "right"
    if left_up and not right_up:
        return "left"
    if left_up and right_up:
        return "both"
    return "none"

from datetime import datetime
def frame_to_llm_string(frame: Dict[str, Any]) -> str:
    ts = frame.get("timestamp_utc")
    dt = datetime.fromisoformat(ts)  # ממיר לאובייקט datetime
    hour_str = dt.strftime("%H:%M:%S")
    hour_str=hour_str[0:5]
    dets = frame.get("detections", [])
    if not dets:
        # אם אין זיהוי—נחזיר מחרוזת "ריקה" שימושית
        return f"state=unknown;time=unknown;Lup=false;Rup=false;bend(kL:unknown,kR:unknown,hL:unknown,hR:unknown);reach=none;conf=low"
    compressor = ""
    person_id = 1
    for det in dets:
        state = det.get("base_state", "unknown")

        feats = det.get("features", {}) or {}
        Lup = bool(feats.get("left_hand_up", False))
        Rup = bool(feats.get("right_hand_up", False))

        kL = _joint_cat(feats.get("knee_l"))
        kR = _joint_cat(feats.get("knee_r"))
        hL = _joint_cat(feats.get("hip_l"))
        hR = _joint_cat(feats.get("hip_r"))

        reach = _reach_from_hands(Lup, Rup)
        conf_num = float(det.get("confidence", {}).get("base_state", 0.0))
        conf = _conf_bin(conf_num)

        if det.get("events"):
            events_str = ",".join(det.get("events"))
        else:
            events_str = "unknown"

            # בניית המחרוזת בפורמט המבוקש
        if len(dets)>1:
            compressor = compressor + (
                f"state={state};"
                f"time={hour_str};"
                f"events={events_str};"
                f"Lup={'true' if Lup else 'false'};"
                f"Rup={'true' if Rup else 'false'};"
                f"bend(kL:{kL},kR:{kR},hL:{hL},hR:{hR});"
                f"reach={reach};"
                f"conf={conf};"
                f"person_id={person_id};"
            ) + ","
            person_id = person_id+1
        else:
          return (
                 f"state={state};"
                 f"time={hour_str};"
                 f"events={events_str};"
                 f"Lup={'true' if Lup else 'false'};"
                 f"Rup={'true' if Rup else 'false'};"
                 f"bend(kL:{kL},kR:{kR},hL:{hL},hR:{hR});"
                 f"reach={reach};"
                 f"conf={conf}"
              )

    return compressor

def compress_motion_json(
    data: Union[str, Dict[str, Any], List[Dict[str, Any]]]
) -> List[str]:
    """
    קלט:
      - path למחרוזת קובץ JSON, או
      - dict עם key בשם "frames": [...], או
      - רשימה של פריימים (list[dict]) או פריים בודד (dict)
    פלט: רשימת מחרוזות קומפקטיות—אחת לכל פריים.
    """
    # אם הגיע path → טען מהדיסק
    if isinstance(data, str):
        with open(data, "r", encoding="utf-8") as f:
            data = json.load(f)

    # נרמל לרשימת פריימים
    if isinstance(data, dict) and "frames" in data:
        frames = data["frames"]
    elif isinstance(data, dict):
        frames = [data]
    elif isinstance(data, list):
        frames = data
    else:
        raise TypeError("Unsupported data type for 'data'")

    return [frame_to_llm_string(fr) for fr in frames]

with open("motion_analysis_room1.json", "r", encoding="utf-8") as f:
    a = json.load(f)

frames = compress_motion_json(a["frames"])
with open("events.txt", "w", encoding="utf-8") as f:
    for s in frames:
        f.write(s + "\n")
