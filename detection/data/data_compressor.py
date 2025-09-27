import json
from typing import Any, Dict, List, Union
from datetime import datetime

# ─────────────────────────── Category Helpers ───────────────────────────
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

# ─────────────────────────── Frame → String Conversion ───────────────────────────
def frame_to_llm_string(frame: Dict[str, Any]) -> str:
    ts = frame.get("timestamp_utc")
    hour_str = "unknown"
    if ts:
        try:
            dt = datetime.fromisoformat(ts)
            hour_str = dt.strftime("%H:%M")  # Truncate to hours:minutes (HH:MM)
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

        # Build the string
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

    # If there were multiple people — return the combined string
    return compressor.rstrip(",")

# ─────────────────────────── General Input → List of Strings ───────────────────────────
def compress_motion_json(
    data: Union[str, Dict[str, Any], List[Dict[str, Any]]]
) -> List[str]:
    """
    Input:
      - path to a JSON file (str), or
      - dict with a key named "frames": [...], or
      - a list of frames (list[dict]) or a single frame (dict)
    Output: a list of compact strings—one per frame.
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

# ─────────────────────────── Export Writing Function ───────────────────────────
def save_compressed_events(
    data_or_path: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    out_path: str,
    file_encoding: str = "utf-8",
) -> int:
    """
    Converts the input (path/obj) into a list of strings and writes them to a text file—one line per frame.
    Returns the number of lines written.
    """

    lines = compress_motion_json(data_or_path)
    with open(out_path, "w", encoding=file_encoding) as f:
        for s in lines:
            f.write(s + "\n")
    print(f"Compressed {data_or_path} with {len(lines)} lines.")

    return len(lines)

