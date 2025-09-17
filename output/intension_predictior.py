import json
from idlelib.iomenu import encoding

import ollama

def intension_generator(model="llama3.2:3b", prompt="", data=""):
    to_model = prompt + "\n" + data
    resp = ollama.generate(model=model, prompt=to_model, options={"temperature":0})
    return resp['response']

with open("../detection/data/events.txt", "r", encoding="utf-8") as f:
    frames = [line.strip() for line in f]

to_remove = "state=unknown;time=unknown;Lup=false;Rup=false;bend(kL:unknown,kR:unknown,hL:unknown,hR:unknown);reach=none;conf=low"
frames = [line.strip() for line in frames if line.strip() != to_remove]
print(frames)

with open("../detection/prompt.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

from tqdm import tqdm
detect = []
import time
for frame in tqdm(frames[:150], desc="Analyzing", unit="frames", mininterval=1):
    #print(frame)
    st = time.time()
    detect.append((intension_generator("llama3.2:3b", prompt, frame)))
    et = time.time()
    #print(f"{et - st:.3f} seconds")

from typing import List, Callable, Dict
import numpy as np

def dedupe_by_embeddings(
    texts: List[str],
    embed_fn: Callable[[List[str]], np.ndarray],
    sim_thresh: float = 0.85
) -> Dict[str, List]:
    """
    Group near-duplicate sentences via cosine similarity on embeddings.
    Keep the shortest sentence as the representative for each cluster.
    Returns {
        "representatives": [
            {"sentence": str, "count": int, "prob": float}, ...
        ]
    }
    """
    if not texts:
        return {"representatives": []}

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
        used[cluster_idxs] = True
        clusters.append(cluster_idxs)

    representatives = []
    total = len(texts)

    for c in clusters:
        rep_sentence = min((texts[j] for j in c), key=lambda s: len(s))
        count = len(c)
        prob = count / total
        representatives.append({
            "sentence": rep_sentence,
            "count": count,
            "prob": prob
        })

    return {"representatives": representatives}

# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_fn(texts: List[str]) -> np.ndarray:
    return np.asarray(_model.encode(texts, convert_to_numpy=True, normalize_embeddings=False))

result = dedupe_by_embeddings(detect, embed_fn, sim_thresh=0.86)
result["origin_detections"] = detect

with open("outputs/result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

