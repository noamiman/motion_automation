# automation_recommender.py
# few-shot using
import os
import json
import re
from collections import OrderedDict
from typing import List, Optional, Dict
import ollama

ALLOWED = {
    "open air-conditioner", "close air-conditioner",
    "turn on lights", "turn off lights",
    "open shutters", "close shutters",
    "turn on heater", "turn off heater",
    "lock door", "unlock door",
    "play music", "stop music",
    "activate sleep mode", "deactivate sleep mode",
    "no automation",
}

ALIAS_PATTERNS = [
    (r"\b(open|turn on)\s+(ac|a/c|air ?conditioner)\b", "open air-conditioner"),
    (r"\b(close|turn off)\s+(ac|a/c|air ?conditioner)\b", "close air-conditioner"),
    (r"\b(turn on|switch on|lights on)\b.*\blight(s)?\b", "turn on lights"),
    (r"\b(turn off|switch off|lights off)\b.*\blight(s)?\b", "turn off lights"),
    (r"\b(open)\b.*\b(shutter|blinds?)\b", "open shutters"),
    (r"\b(close)\b.*\b(shutter|blinds?)\b", "close shutters"),
    (r"\b(turn on|switch on)\b.*\bheater\b", "turn on heater"),
    (r"\b(turn off|switch off)\b.*\bheater\b", "turn off heater"),
    (r"\b(lock)\b.*\bdoor\b", "lock door"),
    (r"\b(unlock)\b.*\bdoor\b", "unlock door"),
    (r"\b(play)\b.*\bmusic\b", "play music"),
    (r"\b(stop|pause)\b.*\bmusic\b", "stop music"),
    (r"\b(activate|enable)\b.*\bsleep mode\b", "activate sleep mode"),
    (r"\b(deactivate|disable)\b.*\bsleep mode\b", "deactivate sleep mode"),
    (r"\bno automation\b", "no automation"),
]

TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")

def _round_down_hour(h: int, m: int) -> str:
    h = max(0, min(23, h))
    return f"{h:02d}:00"

def normalize_recommendation(text: str) -> Optional[str]:
    """Returns '<action> at HH:MM' or None if not recognized."""
    if not text:
        return None
    s = text.strip().lower().strip(" '\"").rstrip(".")
    m = TIME_RE.search(s)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
    else:
        hh, mm = 0, 0
    time_out = _round_down_hour(hh, mm)

    for act in ALLOWED:
        if s.startswith(act):
            return f"{act} at {time_out}"

    for pat, canon in ALIAS_PATTERNS:
        if re.search(pat, s):
            return f"{canon} at {time_out}"

    m2 = re.match(r"\s*([a-z \-]+?)\s+at\s+\d{1,2}:\d{2}\s*$", s)
    if m2:
        action_guess = m2.group(1).strip()
        if action_guess in ALLOWED:
            return f"{action_guess} at {time_out}"
    return None

def intension_generator(model="llama3.2:3b", prompt="", data="") -> str:
    to_model = f"{prompt}\n\nUser input:\n{data}\n\nOutput:"
    resp = ollama.generate(model=model, prompt=to_model, options={"temperature": 0})
    return resp["response"].strip()

def _save_actions_json(actions: List[str], out_path: str, encoding: str = "utf-8") -> None:
    """Saves structured JSON: [{"action": ..., "time": ...}, ...]."""
    structured: List[Dict[str, str]] = []
    for act in actions:
        if " at " in act:
            action, time = act.rsplit(" at ", 1)
            structured.append({"action": action, "time": time})
        else:
            structured.append({"action": act, "time": "00:00"})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding=encoding) as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)

def prompt_generator_by_prob(
    prob: float,
    result_json_path: str,
    prompt_path: str = "prompt_for_auto.txt",
    model_name: str = "llama3.2:3b",
    out_actions_json_path: Optional[str] = None,
) -> List[str]:
    """
    Generates actions from the LLM for candidates above a probability threshold, normalizes, removes duplicates,
    and returns a list. If out_actions_json_path is provided — also saves structured JSON.
    """
    # 1) Read representatives
    with open(result_json_path, "r", encoding="utf-8") as f:
        reps = json.load(f).get("representatives", [])
    sentences = [r["sentence"] for r in reps if float(r.get("prob", 0.0)) >= prob]

    # 2) Load the prompt
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    # 3) LLM → raw recommendations
    raw_recs = [intension_generator(model=model_name, prompt=prompt, data=s) for s in sentences]

    # 4) Normalize + filter None
    normalized = [norm for rec in raw_recs if (norm := normalize_recommendation(rec))]

    # 5) Remove duplicates while preserving order
    deduped = list(OrderedDict.fromkeys(normalized))

    # 6) Optional save to JSON
    if out_actions_json_path:
        _save_actions_json(deduped, out_actions_json_path)

    print(f"Saved automation recommendations in {out_actions_json_path}, include {len(deduped)} options")
    return deduped
