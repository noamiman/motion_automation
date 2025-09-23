from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig


ds = load_dataset("json", data_files={"train": "../yaml_prompts_dataset.jsonl"})
print(ds)  # אמור להראות id, instruction, output_yaml, meta

# סינון רשומות ריקות לפי העמודות הנכונות
ds["train"] = ds["train"].filter(
    lambda x: (x.get("instruction") or "").strip() and (x.get("output_yaml") or "").strip()
)

# אותו טמפלייט כמו באימון, עם גדרות YAML כדי שלא "יברח" באינפרנס
RESPONSE_TEMPLATE = "### Response:\n```yaml\n"
CLOSE_FENCE = "```"

def format_example(ex):
    instr = (ex["instruction"] or "").strip()
    yml   = (ex["output_yaml"] or "").strip()
    return {
        "text": (
            "### Instruction:\n"
            f"{instr}\n\n"
            f"{RESPONSE_TEMPLATE}"
            f"{yml}\n"
            f"{CLOSE_FENCE}\n"
        ),
        # אם אתה רוצה לשמור מזהים לדיבוג, אפשר להשאיר אותם:
        "id": ex.get("id"),
        # "meta": ex.get("meta"),  # רק אם צריך
    }

# אם לא אכפת לך מ-id/meta, מחק הכול; אחרת תמחק רק את העמודות של המקור
ds_train = ds["train"].map(
    format_example,
    remove_columns=ds["train"].column_names  # או: remove_columns=["instruction","output_yaml"]
)

print("rows after map:", ds_train.num_rows)
print(ds_train[0]["text"])  # הצצה


from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
import os

MODEL_ID = "google/gemma-3-1b-pt"  # ודא שיש לך גישה (HF token)

# אופציונלי: התחברות אוטומטית אם יש משתנה סביבה
# from huggingface_hub import login
# if (tok := os.getenv("HUGGINGFACE_HUB_TOKEN")):
#     login(tok)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

# חשוב לג׳מה: לדאוג שיש pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"
tokenizer.model_max_length = 4096

base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    attn_implementation="eager"
)

# חיסכון בזיכרון (אופציונלי אבל מומלץ אם יש עומס):
base.gradient_checkpointing_enable()
base.config.use_cache = False  # חובה יחד עם gradient checkpointing

# LoRA
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"],  # נפוץ לג׳מה
)
model = get_peft_model(base, lora)

# --- קולאטור שמחשב loss רק על מה שאחרי ה-response template ---
RESPONSE_TEMPLATE = "### Response:\n"

class CompletionOnlyCollator:
    def __init__(self, tokenizer, response_template=RESPONSE_TEMPLATE):
        self.tok = tokenizer
        self.templ = response_template
        self.templ_ids = self.tok(self.templ, add_special_tokens=False)["input_ids"]

    def __call__(self, batch):
        import torch
        enc = self.tok([ex["text"] for ex in batch],
                       padding=True, truncation=True, return_tensors="pt")
        labels = enc["input_ids"].clone()

        tlen = len(self.templ_ids)
        templ = torch.tensor(self.templ_ids, dtype=enc["input_ids"].dtype, device=enc["input_ids"].device)

        for i, ids in enumerate(enc["input_ids"]):
            # חפש את תחילת ה-Response
            start = -1
            for j in range(0, ids.size(0) - tlen + 1):
                if torch.equal(ids[j:j+tlen], templ):
                    start = j + tlen
                    break
            # מסך את כל מה שלפני תחילת התשובה
            if start == -1:
                labels[i, :] = -100
            else:
                labels[i, :start] = -100

        enc["labels"] = labels
        return enc

collator = CompletionOnlyCollator(tokenizer, RESPONSE_TEMPLATE)

args = TrainingArguments(
    output_dir="outputs/gemma3-1bpt-yaml-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    remove_unused_columns=False,
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    bf16=torch.cuda.is_available(),  # אם יש
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_train,      # יש רק "text" כמו שבנית
    data_collator=collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model()



#-------------------------

import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

# ====== CONFIG ======
# If you saved LoRA adapters (trainer.save_model()), point to that folder:
ADAPTER_OR_MODEL_DIR = "model_saved_zip/content/outputs/gemma3-1bpt-yaml-lora"   # change if needed
# Base model (needed when loading LoRA adapters)
BASE_MODEL_ID = "google/gemma-3-1b-pt"

# Must match training template:
RESPONSE_TEMPLATE = "### Response:\n```yaml\n"
CLOSE_FENCE = "```"

# ====== DEVICE / DTYPE ======
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else (torch.float16 if DEVICE == "mps" else torch.float32)

def prepare_tokenizer(path_or_id):
    tok = AutoTokenizer.from_pretrained(path_or_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_model_and_tokenizer():
    """
    Auto-detect:
      - If ADAPTER_OR_MODEL_DIR has adapter_model.safetensors → load base + LoRA.
      - Else assume it's a merged full model dir (with model.safetensors).
    """
    p = Path(ADAPTER_OR_MODEL_DIR)
    if (p / "adapter_model.safetensors").exists():
        tok = prepare_tokenizer(BASE_MODEL_ID)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            dtype=DTYPE,
            attn_implementation="eager",  # recommended for Gemma 3
        )
        base.to(DEVICE).eval()
        base.config.use_cache = True
        base.config.pad_token_id = tok.pad_token_id
        model = PeftModel.from_pretrained(base, str(p))
        model.to(DEVICE).eval()
        print("Loaded base model + LoRA adapters.")
        return model, tok
    else:
        # merged full model
        tok = prepare_tokenizer(str(p))
        model = AutoModelForCausalLM.from_pretrained(
            str(p),
            dtype=DTYPE,
            attn_implementation="eager",
        )
        model.to(DEVICE).eval()
        model.config.use_cache = True
        model.config.pad_token_id = tok.pad_token_id
        print("Loaded merged full model.")
        return model, tok

# ====== STOPPING CRITERIA (stop exactly at the closing fence ``` ) ======
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_sequences_ids):
        self.stop_sequences_ids = [torch.tensor(s, dtype=torch.long) for s in stop_sequences_ids]
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for s in self.stop_sequences_ids:
            if input_ids.shape[1] >= len(s) and torch.equal(input_ids[0, -len(s):], s.to(input_ids.device)):
                return True
        return False

def generate_yaml(prompt: str, model, tokenizer, max_new_tokens: int = 256) -> str:
    # Must match training input format
    prefix = f"### Instruction:\n{prompt.strip()}\n\n{RESPONSE_TEMPLATE}"
    inputs = tokenizer(prefix, return_tensors="pt").to(DEVICE)

    # Build stopper for ```
    stop_ids = [tokenizer(CLOSE_FENCE, add_special_tokens=False).input_ids]
    stoppers = StoppingCriteriaList([StopOnTokens(stop_ids)])

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                # greedy = stable YAML
            no_repeat_ngram_size=6,         # avoid loops
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stoppers,     # stop at ```
        )

    # Slice out only the completion and strip the closing fence if present
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.split(CLOSE_FENCE, 1)[0].rstrip()

if __name__ == "__main__":
    model, tok = load_model_and_tokenizer()
    prompt = "Turn on living room lights at 21:00 as a Home Assistant automation"
    yaml_text = generate_yaml(prompt, model, tok, max_new_tokens=256)
    print(yaml_text)

    # Optional: validate YAML (needs pyyaml: pip install pyyaml)
    try:
        import yaml
        yaml.safe_load(yaml_text)
        print("\n(YAML parsed successfully)")
    except Exception as e:
        print("\n[Warning] YAML parse failed:", e)


