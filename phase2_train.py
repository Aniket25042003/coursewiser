# ==============================================================
# phase2_train.py
# ==============================================================
import os
import random
import json
from datetime import datetime
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import PeftModel
import torch

# ==============================================================
# CONFIG
# ==============================================================
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B"  # same base as Phase 1
PHASE1_ADAPTER_DIR = "/kaggle/working/phase1_llama3_3b_lora/lora_adapter"
PHASE1_TOKENIZER_DIR = "/kaggle/working/phase1_llama3_3b_lora"
INSTRUCTION_JSONL = "instruction_dataset.jsonl"
OUTPUT_DIR = "/kaggle/working/phase2_instruct_lora"

NUM_EPOCHS = 2
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 5e-5
MAX_LENGTH = 1024

# ==============================================================
# BitsAndBytes 4-bit Quantization
# ==============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ==============================================================
# Utility: Prompt Formatting
# ==============================================================
def format_prompt(instruction, inp, output):
    if inp and inp.strip():
        return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

# ==============================================================
# Load Dataset
# ==============================================================
records = []
with open(INSTRUCTION_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        inst = obj.get("instruction") or obj.get("question") or ""
        inp = obj.get("input", "")
        out = obj.get("response") or obj.get("output") or obj.get("GPT4_response") or ""
        records.append({"instruction": inst, "input": inp, "output": out})

print(f"Loaded {len(records)} instruction samples.")

random.shuffle(records)
texts = [format_prompt(r["instruction"], r["input"], r["output"]) for r in records]
dataset = Dataset.from_dict({"text": texts})
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = dataset["train"], dataset["test"]
print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

# ==============================================================
# Tokenizer
# ==============================================================
tokenizer = AutoTokenizer.from_pretrained(PHASE1_TOKENIZER_DIR, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(examples):
    toks = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    toks["labels"] = toks["input_ids"].copy()
    return toks

train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ==============================================================
# Load Base + Phase 1 LoRA
# ==============================================================
print("Loading base model (4-bit quantized) and applying Phase 1 adapter...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base, PHASE1_ADAPTER_DIR)
model.print_trainable_parameters()

# ==============================================================
# Training Setup
# ==============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.03,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    save_total_limit=3,
    remove_unused_columns=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# ==============================================================
# Train
# ==============================================================
trainer.train(resume_from_checkpoint=False)

# ==============================================================
# Save Phase 2 Adapter and Tokenizer
# ==============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
trainer.save_state()

# ==============================================================
# Merge Adapter with Base for Deployment
# ==============================================================
print("\nMerging adapter into base model for AWS deployment...")
merged_model_dir = os.path.join(OUTPUT_DIR, "merged_model")
os.makedirs(merged_model_dir, exist_ok=True)

merged_model = model.merge_and_unload()
merged_model.save_pretrained(merged_model_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_model_dir)

# ==============================================================
# Save Metadata for Version Tracking
# ==============================================================
metadata = {
    "base_model": BASE_MODEL_NAME,
    "phase1_adapter": PHASE1_ADAPTER_DIR,
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "num_samples": len(records),
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "grad_accum": GRAD_ACCUM,
    "max_length": MAX_LENGTH,
    "train_size": len(train_ds),
    "val_size": len(val_ds),
}
with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

# ==============================================================
# Summary
# ==============================================================
print("\n✅ Phase 2 Instruction Fine-Tuning Complete!")
print(f"📁 Adapter + Trainer Checkpoints: {OUTPUT_DIR}")
print(f"📦 Deployment-Ready Merged Model: {merged_model_dir}")
print(f"🧾 Metadata Saved: {os.path.join(OUTPUT_DIR, 'training_metadata.json')}")
print("\nTo load for inference on AWS:")
print(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{merged_model_dir}")
model = AutoModelForCausalLM.from_pretrained("{merged_model_dir}", torch_dtype=torch.bfloat16, device_map="auto")

prompt = "Explain reinforcement learning simply."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
""")
