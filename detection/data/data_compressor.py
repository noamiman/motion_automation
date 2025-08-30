import json

import json
from idlelib.iomenu import encoding
from typing import Any, Dict, List, Union

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

def frame_to_llm_string(frame: Dict[str, Any]) -> str:
    dets = frame.get("detections", [])
    if not dets:
        # אם אין זיהוי—נחזיר מחרוזת "ריקה" שימושית
        return "state=unknown;Lup=false;Rup=false;bend(kL:unknown,kR:unknown,hL:unknown,hR:unknown);reach=none;conf=low"

    # נניח detection ראשי ראשון (אם יש כמה אפשר לדרג לפי confidence)
    det = dets[0]
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

    # בניית המחרוזת בפורמט המבוקש
    return (
        f"state={state};"
        f"Lup={'true' if Lup else 'false'};"
        f"Rup={'true' if Rup else 'false'};"
        f"bend(kL:{kL},kR:{kR},hL:{hL},hR:{hR});"
        f"reach={reach};"
        f"conf={conf}"
    )

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

import json

with open("motion_analysis_room1.json", "r", encoding="utf-8") as f:
    a = json.load(f)   # שים לב: load ולא loads


import ollama
def intension_generator(model="llama3.2:3b", prompt="", data=""):
    to_model = prompt + "\n" + data
    resp = ollama.generate(model=model, prompt=to_model, options={"temperature":0})
    return resp['response']

# prompt = "analyze the data and output ONLY a concise, natural-language description of what the person is doing physically, mainly what his intention is. Do not repeat the raw data."
frames = compress_motion_json(a["frames"])
with open("../prompt.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

detect = []
for frame in frames[:20]:
    #print(frame)
    detect.append((intension_generator("llama3.2:3b", prompt, frame)))
print(detect)
#
# prompt2 = "You will receive an array of human detections. analyze every sentence in the array and return a list containing only the detections that are meaningfully different from the others, filtering out those that are similar or redundant. return just a format of array -> []"
# print(intension_generator("llama3.2:3b", prompt2, str(detect)))

import numpy as np
from typing import List, Callable, Dict

def dedupe_by_embeddings(
    texts: List[str],
    embed_fn: Callable[[List[str]], np.ndarray],
    sim_thresh: float = 0.85
) -> Dict[str, List]:
    """
    Group near-duplicate sentences via cosine similarity on embeddings.
    Keep the shortest sentence as the representative for each cluster.
    Returns {"representatives": [...], "clusters": [[idxs...], ...]}
    """
    if not texts:
        return {"representatives": [], "clusters": []}

    X = embed_fn(texts)  # shape: (n, d)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    n = len(texts)
    used = np.zeros(n, dtype=bool)
    clusters = []

    for i in range(n):
        if used[i]:
            continue
        # cosine sim to i
        sims = X @ X[i].T
        cluster_idxs = np.where((sims >= sim_thresh) & (~used))[0].tolist()
        # mark them used
        used[cluster_idxs] = True
        clusters.append(cluster_idxs)

    # representative = shortest text in each cluster
    reps = [min((texts[j] for j in c), key=lambda s: len(s)) for c in clusters]
    return {"representatives": reps, "clusters": clusters}

# Example with sentence-transformers (if installed):
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_fn(texts: List[str]) -> np.ndarray:
    return np.asarray(_model.encode(texts, convert_to_numpy=True, normalize_embeddings=False))

result = dedupe_by_embeddings(detect, embed_fn, sim_thresh=0.86)
print(result["representatives"]) # your “meaningfully different” set

