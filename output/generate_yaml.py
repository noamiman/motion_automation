# output/zero_shot_automation_yaml.py
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Optional, Any

# ====== ניקוי פלט ======

# שורות “דמויות YAML” + תמיכה במפרידי מסמכים
_YAML_LINE = re.compile(
    r"""
    ^\s*$                        # empty
  | ^---\s*$                     # YAML doc start
  | ^\.\.\.\s*$                  # YAML doc end
  | ^\s*#                        # comment
  | ^\s*-\s+.*$                  # list item
  | ^\s*[\w\-\.\'"]+\s*:\s*.*$   # key: value
    """,
    re.VERBOSE,
)

# fence פתיחה סובלני: ```yaml / ```yml / ``yaml / ``yml (רישיות/קטנות)
_FENCE_OPEN_RE  = re.compile(r"^\s*`{2,3}\s*y(?:a)?ml\s*$", re.I)
_FENCE_CLOSE_RE = re.compile(r"^\s*`{3}\s*$")

def _clean_yaml(text: str) -> str:
    """מסיר פתיחי fence, חותך ב־fence סגירה, ומשאיר רק שורות שנראות YAML."""
    s = (text or "").replace("\r\n", "\n").lstrip()
    lines_in = s.splitlines()

    # הסרת fence פתיחה אם השורה הראשונה היא fence
    if lines_in and _FENCE_OPEN_RE.match(lines_in[0]):
        lines_in = lines_in[1:]

    cleaned: List[str] = []
    started = False
    for ln in lines_in:
        # עצירה על fence סגירה ```
        if _FENCE_CLOSE_RE.match(ln):
            break
        if not started and not ln.strip():
            continue
        if _YAML_LINE.match(ln):
            started = True
            cleaned.append(ln)

    return "\n".join(cleaned).strip()


# ====== נרמול HA ======
try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

_DOMAIN_PLACEHOLDER = {
    "cover":   "cover.living_room",
    "light":   "light.living_room",
    "lock":    "lock.front_door",
    "climate": "climate.living_room_ac",
}

def _infer_domain(service: str) -> Optional[str]:
    if not isinstance(service, str) or "." not in service:
        return None
    return service.split(".", 1)[0]

def _normalize_ha_yaml(ytext: str) -> str:
    """
    תיקונים עדינים:
      - אם condition חסר/ריק → []
      - action תמיד רשימה; לכל action ודא target.entity_id (placeholder לפי domain אם חסר)
      - mode: single
      - אם הוחזר list של אוטומציות — קח את הראשונה (המודל אמור להחזיר אחת)
    """
    if not ytext.strip() or not _HAS_YAML:
        return ytext.strip()

    try:
        data = yaml.safe_load(ytext)
    except Exception:
        return ytext.strip()

    # אם הוחזרה רשימה של מסמכים — קח את הראשון התקין
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                data = item
                break

    if not isinstance(data, dict):
        return ytext.strip()

    # condition
    cond = data.get("condition", [])
    if not cond:
        data["condition"] = []

    # action כ-list + הוספת target.entity_id
    acts = data.get("action", [])
    if isinstance(acts, dict):
        acts = [acts]
    if not isinstance(acts, list):
        acts = []

    fixed: List[dict] = []
    for a in acts:
        if not isinstance(a, dict):
            continue
        svc = a.get("service")
        dom = _infer_domain(svc) if svc else None

        tgt = a.get("target", {})
        if not isinstance(tgt, dict):
            tgt = {}
        if "entity_id" not in tgt or not str(tgt.get("entity_id", "")).strip():
            if dom in _DOMAIN_PLACEHOLDER:
                tgt["entity_id"] = _DOMAIN_PLACEHOLDER[dom]
        if tgt:
            a["target"] = tgt
        fixed.append(a)

    if fixed:
        data["action"] = fixed

    # mode
    data["mode"] = "single"

    try:
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True).strip()
    except Exception:
        return ytext.strip()


# ====== המחלקה הראשית ======
class ZeroShotAutomationYAML:
    """
    יצירת YAML לאוטומציות Home Assistant ב-zero-shot דרך Ollama.
    כולל ניקוי פלט + נרמול ל-HA.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a Home Assistant automation YAML generator.

TASK
Given a short natural-language instruction, output ONE valid Home Assistant automation as YAML.

HARD RULES
- Output ONLY a fenced YAML block and nothing else.
- Top-level keys exactly: id, alias, trigger, condition, action, mode.
- Time like "at HH:MM" -> trigger: platform: time, at: "HH:MM:00".
- If the instruction does NOT mention any condition, set `condition: []`. DO NOT invent conditions.
- Every action MUST include `target.entity_id`. If missing, use a sensible placeholder:
  * cover.open_cover / cover.close_cover -> cover.living_room
  * light.turn_on / light.turn_off -> light.living_room
  * lock.lock / lock.unlock -> lock.front_door
  * climate.turn_on / climate.set_hvac_mode -> climate.living_room_ac
- Use correct services (no made-up domains).
- Use `mode: single`.
- No prose, no comments, no extra keys.
"""

    def __init__(self, model: str = "llama3.2:3b", system_prompt_path: Optional[str] = None) -> None:
        try:
            import ollama  # type: ignore
        except ImportError as e:
            raise ImportError("Missing dependency: pip install ollama") from e

        self._ollama = ollama
        self.model = model

        if system_prompt_path and Path(system_prompt_path).exists():
            self.system_prompt = Path(system_prompt_path).read_text(encoding="utf-8").rstrip()
        else:
            self.system_prompt = self.DEFAULT_SYSTEM_PROMPT.rstrip()

    # --- עזר פנימי: הרצה + ניקוי + נרמול ---
    def _generate_one(self, user_text: str, temperature: float = 0.0, max_new_tokens: int = 200) -> str:
        """
        בונה prompt יציב (כולל fence פתיחה), מריץ Ollama, מנקה ונורמל ל-HA.
        """
        prompt = (
            f"{self.system_prompt}\n"
            f"INPUT:\n{user_text.strip()}\n\n"
            "OUTPUT (YAML only)\n```yaml\n"
        )
        resp = self._ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": temperature,
                "repeat_penalty": 1.1,
                "num_predict": max_new_tokens,
                "stop": ["```"],  # עצור על גדר סגירה
            },
        )
        raw = (resp.get("response") or "")
        cleaned = _clean_yaml(raw)

        # fallback: נסה לחלץ בין fences אם הניקוי החזיר ריק
        if not cleaned:
            m = re.search(r"```y?a?ml\s*(.*?)```", raw, flags=re.S | re.I)
            if m:
                cleaned = _clean_yaml(m.group(1))

        return _normalize_ha_yaml(cleaned)

    @staticmethod
    def _build_prompts_from_actions(actions: Any) -> List[str]:
        """
        מקבל:
          - [{"action": "...", "time": "HH:MM"}, ...]  או  ["turn on lights at 21:00", ...]
        ומחזיר מחרוזות zero-shot בסגנון:
          "Create an automation that <action> at <time> with Home Assistant"
        """
        if not isinstance(actions, list):
            raise TypeError("actions JSON must be a list (dicts or strings).")

        prompts: List[str] = []
        for item in actions:
            if isinstance(item, dict):
                act = str(item.get("action", "")).strip()
                t   = str(item.get("time", "")).strip()
                if act and t:
                    prompts.append(f"Create an automation that {act} at {t} with Home Assistant")
                elif act:
                    prompts.append(f"Create an automation that {act} with Home Assistant")
            elif isinstance(item, str):
                s = item.strip()
                if s:
                    prompts.append(f"Create an automation that {s} with Home Assistant")
        return prompts

    # --- API ראשי ---
    def run(
        self,
        actions_json_path: str,
        out_path: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 200,
    ) -> List[str]:
        """
        קורא את actions_json_path, בונה פרומפטים, מפיק YAML (נקי ומנורמל), ומחזיר רשימה.
        אם out_path סופק – כותב קובץ רב-מסמכי (מופרד ב־'---').
        """
        with open(actions_json_path, "r", encoding="utf-8") as f:
            actions = json.load(f)

        prompts = self._build_prompts_from_actions(actions)

        outputs: List[str] = []
        for p in prompts:
            txt = self._generate_one(p, temperature=temperature, max_new_tokens=max_new_tokens)
            if txt:
                outputs.append(txt)

        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as g:
                for i, y in enumerate(outputs):
                    if i > 0:
                        g.write("\n---\n")
                    g.write(y.strip() + "\n")

        return outputs



# ====== פונקציית עוטפת לשימוש נוח מה-main ======
def generate_automation_yaml_zeroshot(
    actions_json: str,
    prompt_path: Optional[str],
    model_name: str = "llama3.2:3b",
    out_yaml_path: Optional[str] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 200,
) -> List[str]:
    """
    עטיפה נוחה: מייצר YAML ב-zero-shot דרך Ollama, כולל ניקוי ונרמול.
    """
    gen = ZeroShotAutomationYAML(model=model_name, system_prompt_path=prompt_path)
    return gen.run(
        actions_json_path=actions_json,
        out_path=out_yaml_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
