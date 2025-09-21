# intent_pipeline.py
import json
import time
from typing import Any, Dict, List, Callable, Optional

import numpy as np
from tqdm import tqdm
import ollama
from sentence_transformers import SentenceTransformer

# ─────────────────────────── IO: קריאה ושמירה ───────────────────────────
def load_frames_txt(path: str) -> List[str]:
    """טוען שורות (פריימים) מקובץ טקסט ומסנן את שורת ברירת־המחדל הריקה."""
    to_remove = ("state=unknown;time=unknown;Lup=false;Rup=false;"
                 "bend(kL:unknown,kR:unknown,hL:unknown,hR:unknown);"
                 "reach=none;conf=low")
    with open(path, "r", encoding="utf-8") as f:
        frames = [ln.strip() for ln in f]
    return [ln for ln in frames if ln and ln != to_remove]

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_result_json(result: Dict[str, Any], out_path: str, ensure_pretty: bool = True) -> None:
    """פונקציה יעודית לכתיבת תוצאות ל־JSON (שורה שניתן לייבא ולהשתמש בה)."""
    kwargs = {"ensure_ascii": False}
    if ensure_pretty:
        kwargs["indent"] = 2
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, **kwargs)

# ─────────────────────────── LLM ───────────────────────────
def intension_generator(model: str, prompt: str, data: str, temperature: float = 0.0) -> str:
    to_model = f"{prompt}\n{data}"
    resp = ollama.generate(model=model, prompt=to_model, options={"temperature": temperature})
    return resp["response"]

def run_llm_over_frames(
    frames: List[str],
    prompt: str,
    model_name: str = "llama3.2:3b",
    limit: Optional[int] = 150,
    show_progress: bool = True,
) -> List[str]:
    """מריץ את המודל על כל פריים ומחזיר רשימת טקסטים (detect)."""
    if limit is not None:
        frames = frames[:limit]

    detects: List[str] = []
    iterator = tqdm(frames, desc="Analyzing", unit="frames", mininterval=1) if show_progress else frames
    for frame in iterator:
        _ = time.time()
        out = intension_generator(model_name, prompt, frame)
        detects.append(out)
    return detects

# ─────────────────────────── אמבדינגים + דה-דופליקציה ───────────────────────────
# נטען פעם אחת (cache)
_EMB_MODEL: Optional[SentenceTransformer] = None

def get_embed_model() -> SentenceTransformer:
    global _EMB_MODEL
    if _EMB_MODEL is None:
        _EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMB_MODEL

def embed_fn(texts: List[str]) -> np.ndarray:
    model = get_embed_model()
    return np.asarray(model.encode(texts, convert_to_numpy=True, normalize_embeddings=False))

def dedupe_by_embeddings(
    texts: List[str],
    embed_func: Callable[[List[str]], np.ndarray],
    sim_thresh: float = 0.86
) -> Dict[str, List[Dict[str, Any]]]:
    """
    קיבוץ משפטים דומים לפי קוסיין-סימילריטי על אמבדינגים.
    לכל קלאסטר נשמור את *המשפט הקצר ביותר* כנציג.
    """
    if not texts:
        return {"representatives": []}

    X = embed_func(texts)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    n = len(texts)
    used = np.zeros(n, dtype=bool)
    clusters: List[List[int]] = []

    for i in range(n):
        if used[i]:
            continue
        sims = X @ X[i].T
        cluster_idxs = np.where((sims >= sim_thresh) & (~used))[0].tolist()
        used[cluster_idxs] = True
        clusters.append(cluster_idxs)

    representatives: List[Dict[str, Any]] = []
    total = len(texts)
    for c in clusters:
        rep_sentence = min((texts[j] for j in c), key=len)
        count = len(c)
        prob = count / total
        representatives.append({"sentence": rep_sentence, "count": count, "prob": prob})

    return {"representatives": representatives}

# ─────────────────────────── פונקציית-מעטפת שמייצרת ושומרת תוצאה ───────────────────────────
def build_result_dict(
    frames_txt_path: str,
    prompt_path: str,
    model_name: str = "llama3.2:3b",
    limit: Optional[int] = 150,
    sim_thresh: float = 0.86,
) -> Dict[str, Any]:
    """
    בונה את אובייקט התוצאה המלא:
    - קורא פריימים
    - מריץ LLM
    - מדלל דופליקטים עם אמבדינגים
    - מחזיר dict מוכן לכתיבה
    """
    frames = load_frames_txt(frames_txt_path)
    prompt = load_prompt(prompt_path)
    detect = run_llm_over_frames(frames, prompt, model_name=model_name, limit=limit)
    dedup = dedupe_by_embeddings(detect, embed_fn, sim_thresh=sim_thresh)
    dedup["origin_detections"] = detect
    return dedup

import os
from typing import Optional

def write_result(
    frames_txt_path: str,
    prompt_path: str,
    out_json_path: str = "outputs/result.json",
    model_name: str = "llama3.2:3b",
    limit: Optional[int] = 150,
    sim_thresh: float = 0.86,
) -> None:
    """
    הפעולה המבוקשת: בונה את התוצאה ושומרת אותה ל־JSON.
    אם התיקייה של out_json_path לא קיימת – תיווצר אוטומטית.
    """
    result = build_result_dict(
        frames_txt_path=frames_txt_path,
        prompt_path=prompt_path,
        model_name=model_name,
        limit=limit,
        sim_thresh=sim_thresh,
    )

    # יצירת התיקייה אם לא קיימת
    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)

    save_result_json(result, out_json_path)
    print(f"Predict USER intention to {out_json_path} by {model_name}")
