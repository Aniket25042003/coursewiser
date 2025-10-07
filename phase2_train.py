# phase2_instruction_finetune_fixed.py
import os
import random
import json
from datasets import Dataset
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

# -------------------------
# CONFIG
# -------------------------
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B"
PHASE1_ADAPTER_DIR ="./phase1_llama3_3b_lora\lora_adapter"
PHASE1_TOKENIZER_DIR = "./phase1_llama3_3b_lora"
INSTRUCTION_JSONL = "merged_instruction_dataset.jsonl"
OUTPUT_DIR = "./phase2_instruct_lora"

NUM_EPOCHS = 5
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 5e-5
MAX_LENGTH = 1024

# -------------------------
# BitsAndBytes config
# -------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# -------------------------
# Helper: Format prompt
# -------------------------
def format_prompt(instruction, inp, output):
    if inp and inp.strip():
        return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

# -------------------------
# Load dataset
# -------------------------
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
train_ds = dataset["train"]
val_ds = dataset["test"]
print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

# -------------------------
# Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(PHASE1_TOKENIZER_DIR, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(examples):
    toks = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    toks["labels"] = toks["input_ids"].copy()
    return toks

train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# -------------------------
# Load model + LoRA
# -------------------------
print("Loading base model (quantized) and applying Phase 1 adapter...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(base, PHASE1_ADAPTER_DIR)

# ✅ Enable trainable LoRA parameters
model.train()
for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True

model.print_trainable_parameters()

# -------------------------
# Training setup
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.03,
    fp16=True,
    eval_strategy="epoch",  # ✅ new argument name (was evaluation_strategy)
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

# -------------------------
# Train
# -------------------------
trainer.train(resume_from_checkpoint=False)

# -------------------------
# Save model + metadata
# -------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Saving LoRA adapter and tokenizer...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save metadata for reproducibility
metadata = {
    "base_model": BASE_MODEL_NAME,
    "phase1_adapter": PHASE1_ADAPTER_DIR,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "lr": LR,
    "train_samples": len(train_ds),
    "val_samples": len(val_ds),
    "max_length": MAX_LENGTH,
}
with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Phase 2 instruction fine-tuning complete.")
print(f"Model adapter and tokenizer saved at: {OUTPUT_DIR}")
