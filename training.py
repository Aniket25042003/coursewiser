#!/usr/bin/env python3
"""
train_phaseA_qlora_gemma2b.py

Phase A: Domain-absorption training (QLoRA) on Gemma 2B.

Inputs:
- JSONL dataset with one JSON object per line having a "text" field,
  e.g. {"text": "<bos>\n...chunk...\n<eos>"}

Outputs:
- LoRA checkpoint directory (PEFT format)
- Optionally merged model for inference
"""

import os
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import BitsAndBytesConfig

# -------------------------
# USER CONFIG
# -------------------------
MODEL_ID = "google/gemma-2b"   # Hugging Face model id (you may need to accept terms)
DATA_JSONL = "pretrain/chapter11_pretrain.jsonl"  # path to your JSONL
OUTPUT_DIR = "outputs/gemma2b_qlora_phaseA"
LOGGING_DIR = "runs/gemma2b_qlora_phaseA"

# QLoRA / LoRA hyperparams (good defaults; tune if you have more/less VRAM)
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj", "wi", "wo"]  # common sets; may vary by model architecture

# training
BATCH_SIZE = 8              # per device train batch (effective batch = BATCH_SIZE * gradient_accumulation_steps)
GRAD_ACCUM = 4
EPOCHS = 3
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 100
MAX_SEQ_LEN = 1024          # adapt depending on your chunks; you used ~800 chars
EVAL_STEPS = 500
SAVE_STEPS = 500
SEED = 42

# bitsandbytes 4-bit config (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",        # NF4 recommended for QLoRA
    bnb_4bit_compute_dtype=torch.float16
)

# -------------------------
# Setup & utils
# -------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGGING_DIR, exist_ok=True)

# -------------------------
# Load dataset
# -------------------------
# Expects each JSONL line to contain {"text": "..."} (you already have this format)
dataset = load_dataset("json", data_files={"train": DATA_JSONL})
# optional: small val split
dataset = dataset["train"].train_test_split(test_size=0.05, seed=SEED)
train_ds = dataset["train"]
eval_ds = dataset["test"]

# -------------------------
# Tokenizer
# -------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
# make sure tokenizer has pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

# -------------------------
# Tokenize
# -------------------------
def tokenize_batch(examples):
    # input is the "text" field with <bos> and <eos> markers
    texts = examples["text"]
    # We do simple truncation/padding
    out = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_tensors=None,
    )
    # For causal LM labels are the input ids
    out["labels"] = out["input_ids"].copy()
    return out

print("Tokenizing dataset...")
train_ds = train_ds.map(tokenize_batch, batched=True, remove_columns=["text"], num_proc=1)
eval_ds = eval_ds.map(tokenize_batch, batched=True, remove_columns=["text"], num_proc=1)

# Data collator - for causal LM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -------------------------
# Load model in 4-bit and prepare for PEFT
# -------------------------
print("Loading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",   # let accelerate / transformers set optimal device mapping
    trust_remote_code=True,  # gemma may require remote code
)

# resize token embeddings if we added pad token
model.resize_token_embeddings(len(tokenizer))

# Convert model to be prepared for int8/4bit training (fixes some layers/biases)
# Note: prepare_model_for_int8_training is designed for int8 flows; it is commonly used
# as a helper prior to applying PEFT (LoRA). It also sets requires_grad appropriately.
model = prepare_model_for_int8_training(model)

# -------------------------
# Setup LoRA (PEFT)
# -------------------------
print("Attaching LoRA adapters...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# -------------------------
# Training args
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    fp16=True,  # use fp16 mixed precision for speed
    optim="paged_adamw_32bit",  # uses optimizer that works well with bnb
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    run_name="gemma2b-qlora-phaseA",
    remove_unused_columns=False,
    report_to="none",  # set to "wandb" if you configured W&B
)

# -------------------------
# Trainer
# -------------------------
print("Preparing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# -------------------------
# Train
# -------------------------
print("Starting training...")
trainer.train()
print("Training finished.")

# -------------------------
# Save trainer final state (resume-able)
# -------------------------
print("Saving final Trainer state + tokenizer...")
final_trainer_dir = os.path.join(OUTPUT_DIR, "trainer_final")
os.makedirs(final_trainer_dir, exist_ok=True)
trainer.save_model(final_trainer_dir)  # saves model + optimizer/scheduler state
tokenizer.save_pretrained(final_trainer_dir)
print(f"✅ Trainer state saved at: {final_trainer_dir}")

# -------------------------
# Save only PEFT adapters (small LoRA checkpoint)
# -------------------------
print("Saving LoRA adapter checkpoint...")
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
print(f"✅ LoRA adapters saved at: {os.path.join(OUTPUT_DIR, 'lora_adapter')}")

# -------------------------
# Merge LoRA into base model and save full merged checkpoint
# -------------------------
from peft import AutoPeftModelForCausalLM

print("Merging LoRA adapters into base model...")
peft_model = AutoPeftModelForCausalLM.from_pretrained(
    os.path.join(OUTPUT_DIR, "lora_adapter"),
    low_cpu_mem_usage=True,
    device_map="auto",
)

# Merge and unload to get a clean base model with LoRA applied
merged_model = peft_model.merge_and_unload()

# Save merged model (ready for Phase B fine-tuning or deployment)
MERGED_DIR = os.path.join(OUTPUT_DIR, "merged_model")
os.makedirs(MERGED_DIR, exist_ok=True)
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)

print(f"✅ Merged model saved at: {MERGED_DIR}")

