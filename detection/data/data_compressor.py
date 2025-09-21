import json
from typing import Any, Dict, List, Union
from datetime import datetime

# ─────────────────────────── עזרי קטגוריות ───────────────────────────
def _joint_cat(val: Any) -> str:
    if val is None:
        return "unknown"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "unknown"
    if v < 10:
        return "straight"
    if v < 40:
        return "slight_bend"
    return "deep_bend"

def _conf_bin(p: float) -> str:
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

# ─────────────────────────── המרה לפריים → מחרוזת ───────────────────────────
def frame_to_llm_string(frame: Dict[str, Any]) -> str:
    ts = frame.get("timestamp_utc")
    hour_str = "unknown"
    if ts:
        try:
            dt = datetime.fromisoformat(ts)
            hour_str = dt.strftime("%H:%M")  # חותכים לשעות:דקות
        except Exception:
            pass

    dets = frame.get("detections", [])
    if not dets:
        return (
            "state=unknown;time=unknown;Lup=false;Rup=false;"
            "bend(kL:unknown,kR:unknown,hL:unknown,hR:unknown);"
            "reach=none;conf=low"
        )

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
        conf_num = float((det.get("confidence", {}) or {}).get("base_state", 0.0))
        conf = _conf_bin(conf_num)

        events_list = det.get("events")
        events_str = ",".join(events_list) if events_list else "unknown"

        # בניית המחרוזת
        if len(dets) > 1:
            compressor += (
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
            person_id += 1
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

    # אם היו כמה אנשים — מחזירים את המחרוזת המשולבת
    return compressor.rstrip(",")

# ─────────────────────────── המרת קלט כללי לרשימת מחרוזות ───────────────────────────
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
    if isinstance(data, str):
        with open(data, "r", encoding="utf-8") as f:
            data = json.load(f)

    if isinstance(data, dict) and "frames" in data:
        frames = data["frames"]
    elif isinstance(data, dict):
        frames = [data]
    elif isinstance(data, list):
        frames = data
    else:
        raise TypeError("Unsupported data type for 'data'")

    return [frame_to_llm_string(fr) for fr in frames]

# ─────────────────────────── פונקציית כתיבה לייצוא ───────────────────────────
def save_compressed_events(
    data_or_path: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    out_path: str,
    file_encoding: str = "utf-8",
) -> int:
    """
    ממיר את הקלט (path/obj) לרשימת מחרוזות ושומר לקובץ טקסט—שורה לכל פריים.
    מחזיר את מספר השורות שנכתבו.
    """
    lines = compress_motion_json(data_or_path)
    with open(out_path, "w", encoding=file_encoding) as f:
        for s in lines:
            f.write(s + "\n")
    print(f"Compressed {data_or_path} with {len(lines)} lines.")

    return len(lines)

# # אופציונלי: הרצה כ־CLI פשוט
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Compress motion JSON and save to txt.")
#     parser.add_argument("input", help="Path to JSON file or directory-like data")
#     parser.add_argument("output", help="Path to output txt file")
#     args = parser.parse_args()
#     n = save_compressed_events(args.input, args.output)
#     print(f"Wrote {n} lines to {args.output}")
