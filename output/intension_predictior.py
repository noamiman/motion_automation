import json
import time
from typing import Any, Dict, List, Callable, Optional
import numpy as np
from tqdm import tqdm
import ollama
from sentence_transformers import SentenceTransformer
import os

def load_frames_txt(path: str) -> List[str]:
    """Loads lines (frames) from a text file and filters out the default blank line."""
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
    """Dedicated function for writing results to JSON (as a single, importable line)."""
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
    """Runs the model on each frame and returns a list of texts (detect)."""
    if limit is not None:
        frames = frames[:limit]

    detects: List[str] = []
    iterator = tqdm(frames, desc="Analyzing", unit="frames", mininterval=1) if show_progress else frames
    for frame in iterator:
        _ = time.time()
        out = intension_generator(model_name, prompt, frame)
        detects.append(out)
    return detects

# ─────────────────────────── Embeddings + Deduplication ───────────────────────────
# Load once (cache)
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
    Group similar sentences by cosine similarity over embeddings.
    For each cluster, keep the *shortest sentence* as the representative.
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

# ─────────────────────────── Wrapper function that generates and saves a result ───────────────────────────
def build_result_dict(
    frames_txt_path: str,
    prompt_path: str,
    model_name: str = "llama3.2:3b",
    limit: Optional[int] = 150,
    sim_thresh: float = 0.86,
) -> Dict[str, Any]:
    """
    Constructs the complete result object:
    - Reads frames
    - Runs the LLM
    - Deduplicates via embeddings
    - Returns a dict ready for writing
    """

    frames = load_frames_txt(frames_txt_path)
    prompt = load_prompt(prompt_path)
    detect = run_llm_over_frames(frames, prompt, model_name=model_name, limit=limit)
    dedup = dedupe_by_embeddings(detect, embed_fn, sim_thresh=sim_thresh)
    dedup["origin_detections"] = detect
    return dedup


def write_result(
    frames_txt_path: str,
    prompt_path: str,
    out_json_path: str = "outputs/representatives_result.json",
    model_name: str = "llama3.2:3b",
    limit: Optional[int] = 150,
    sim_thresh: float = 0.86,
) -> None:
    """
    The requested operation: builds the result and saves it to JSON.
    If the directory for out_json_path doesn't exist, it will be created automatically.
    """

    result = build_result_dict(
        frames_txt_path=frames_txt_path,
        prompt_path=prompt_path,
        model_name=model_name,
        limit=limit,
        sim_thresh=sim_thresh,
    )

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)

    save_result_json(result, out_json_path)
    print(f"Predict USER intention to {out_json_path} by {model_name}")
