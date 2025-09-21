from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig

ds = load_dataset("json", data_files={"train": "../yaml_prompts_dataset.jsonl"})
print(ds)

def format_example(ex):
    # מגדיר מרקר קבוע לתשובה כדי שנוכל למסך loss על החלק של ה-YAML בלבד
    return {
        "text": (
            "### Instruction:\n"
            f"{ex['prompt'].strip()}\n\n"
            "### Response:\n"  # ← זהו ה-response_template
            f"{ex['yaml'].strip()}\n"
        )
    }

ds_train = ds["train"].map(format_example, remove_columns=ds["train"].column_names)
# עכשיו יש עמודה אחת: text
print(ds_train)

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

collator = CompletionOnlyCollator(tokenizer)

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

