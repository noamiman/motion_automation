import json
from idlelib.iomenu import encoding
import ollama
from anyio import open_signal_receiver
import re
from collections import OrderedDict

ALLOWED = {
    "open air-conditioner",
    "close air-conditioner",
    "turn on lights",
    "turn off lights",
    "open shutters",
    "close shutters",
    "turn on heater",
    "turn off heater",
    "lock door",
    "unlock door",
    "play music",
    "stop music",
    "activate sleep mode",
    "deactivate sleep mode",
    "no automation",
}

# simple alias patterns -> canonical action
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

def normalize_recommendation(text: str) -> str | None:
    """Return canonical '<action> at HH:MM' or None if cannot parse."""
    if not text:
        return None
    s = text.strip().lower()
    # remove surrounding quotes and trailing punctuation
    s = s.strip(" '\"").rstrip(".")
    # find time (default 00:00 if missing)
    m = TIME_RE.search(s)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
    else:
        hh, mm = 0, 0
    time_out = _round_down_hour(hh, mm)

    # try direct allowed phrase first (e.g., 'turn on lights at 21:00')
    for act in ALLOWED:
        if s.startswith(act):
            return f"{act} at {time_out}"

    # otherwise map via aliases
    for pat, canon in ALIAS_PATTERNS:
        if re.search(pat, s):
            return f"{canon} at {time_out}"

    # fallback: try to extract '... at HH:MM' action text before ' at '
    m2 = re.match(r"\s*([a-z \-]+?)\s+at\s+\d{1,2}:\d{2}\s*$", s)
    if m2:
        action_guess = m2.group(1).strip()
        if action_guess in ALLOWED:
            return f"{action_guess} at {time_out}"

    # give up
    return None

def intension_generator(model="llama3.2:3b", prompt="", data=""):
    to_model = f"{prompt}\n\nUser input:\n{data}\n\nOutput:"
    resp = ollama.generate(
        model=model,
        prompt=to_model,
        options={"temperature": 0}
    )
    return resp['response'].strip()

def prompt_generator_by_prob(prob: float, data_loc: str) -> list[str]:
    with open(data_loc, "r", encoding='utf-8') as f:
        res = json.load(f)["representatives"]

    probs_sentences = [r["sentence"] for r in res if float(r["prob"]) >= prob]

    with open("prompt_for_auto.txt", "r", encoding='utf-8') as f:
        prompt = f.read()

    raw_recs = [
        intension_generator(model="llama3.2:3b", prompt=prompt, data=sent)
        for sent in probs_sentences
    ]

    # normalize + drop Nones
    normalized = [norm for rec in raw_recs if (norm := normalize_recommendation(rec))]

    # dedupe while preserving order
    deduped = list(OrderedDict.fromkeys(normalized))

    return deduped


print(prompt_generator_by_prob(0.1, "outputs/result.json"))

