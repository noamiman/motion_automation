# main.py
import os
import sys
from typing import List, Optional, Any
from torch.backends.quantized import engine

from detection.video_output import MotionAnalyzer
from detection.data.data_compressor import save_compressed_events
from output.intension_predictior import write_result
from output.automation_prompt_generator import prompt_generator_by_prob
from output.generate_yaml import ZeroShotAutomationYAML

# ---- הגדרות מרכזיות ----
MODEL_PATH = "yolov8n-pose.pt"
VIDEO_SOURCE = 0  # 0 = מצלמה, או לשים path לקובץ וידאו

MOTION_JSON = "detection/data/motion_analysis_room1.json"
EVENTS_TXT = "detection/data/events.txt"

RESULT_JSON = "output/outputs/result.json"
ACTIONS_JSON = "output/outputs/automation_rec.json"
PROMPT_PATH = "detection/prompt.txt"
PROMPT_AUTO_PATH = "output/prompt_for_auto.txt"

MODEL_NAME = "llama3.2:3b"
LIMIT_DETECTIONS = 100
SIM_THRESH = 0.86
ACTIONS_MIN_PROB = 0.10

ACTIONS_JSON = "output/outputs/automation_rec.json"
OUT_YAML     = "output/outputs/yaml_file_result.yaml"
PROMPT_ZEROSHOT_PATH = "output/prompt_for_zeroShot"


def analyze_motion(model_path: str, video_source, output_json: str, show: bool = True) -> None:
    """שלב 1: לכידת וידאו/מצלמה + ניתוח תנועה → JSON."""
    analyzer = MotionAnalyzer(
        model_path=model_path,
        video_source=video_source,
        output_json=output_json,
        show=show,
    )
    analyzer.run()


def compress_events(input_json: str, out_txt: str) -> int:
    """שלב 2: דחיסת JSON לשורות קומפקטיות → TXT."""
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    n = save_compressed_events(input_json, out_txt)
    #print(f"[compress_events] wrote {n} lines → {out_txt}")
    return n


def predict_intentions(frames_txt: str, prompt_path: str, out_json: str,
                       model_name: str, limit: int | None, sim_thresh: float) -> None:
    """שלב 3: הרצת LLM על האירועים ויצירת result.json."""
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    write_result(
        frames_txt_path=frames_txt,
        prompt_path=prompt_path,
        out_json_path=out_json,
        model_name=model_name,
        limit=limit,
        sim_thresh=sim_thresh,
    )
    #print(f"[predict_intentions] wrote: {out_json}")


def generate_automation_recs(result_json: str, prompt_path: str, out_actions_json: str,
                             model_name: str, prob: float) -> list[str]:
    """שלב 4: יצירת המלצות אוטומציה ושמירתן כ-JSON; מחזיר גם את הרשימה."""
    actions = prompt_generator_by_prob(
        prob=prob,
        result_json_path=result_json,
        prompt_path=prompt_path,
        model_name=model_name,
        out_actions_json_path=out_actions_json,  # שמירה אוטומטית בתוך הפונקציה
    )
    #print(f"[generate_automation_recs] {len(actions)} actions → {out_actions_json}")
    return actions

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



def main() -> None:
    try:
        analyze_motion(MODEL_PATH, VIDEO_SOURCE, MOTION_JSON, show=True)
        compress_events(MOTION_JSON, EVENTS_TXT)
        predict_intentions(
            frames_txt=EVENTS_TXT,
            prompt_path=PROMPT_PATH,
            out_json=RESULT_JSON,
            model_name=MODEL_NAME,
            limit=LIMIT_DETECTIONS,
            sim_thresh=SIM_THRESH,
        )
        actions = generate_automation_recs(
            result_json=RESULT_JSON,
            prompt_path=PROMPT_AUTO_PATH,
            out_actions_json=ACTIONS_JSON,
            model_name=MODEL_NAME,
            prob=ACTIONS_MIN_PROB,
        )

        yamls = generate_automation_yaml_zeroshot(
            actions_json=ACTIONS_JSON,
            prompt_path=PROMPT_ZEROSHOT_PATH,  # או None כדי להשתמש בדיפולט
            model_name=MODEL_NAME,
            out_yaml_path=OUT_YAML,
            temperature=0.0,
            max_new_tokens=200,
        )
        print(f"Wrote {len(yamls)} YAML automation(s) → {OUT_YAML}")

        # הדפסה אופציונלית של הפעולות עצמן:
        for a in actions:
            print("  •", a)

    except KeyboardInterrupt:
        print("\n[main] interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"[main] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
