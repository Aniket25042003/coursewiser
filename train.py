# ======================================
# Phase 1: QLoRA Fine-tuning (Absorption Phase)
# Model: LLaMA 3.2 3B
# ======================================

# !pip install -q bitsandbytes transformers peft accelerate datasets trl

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch, os

# ===========================
# Configurable Parameters
# ===========================

MODEL_NAME = "meta-llama/Llama-3.2-3B"  # ✅ Small, fast, long-context model
DATA_PATH = "dataset.jsonl"             # Path to your preprocessed dataset
OUTPUT_DIR = "./phase1_llama3_3b_lora"  # Folder to save model checkpoints

# Training parameters
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LR = 2e-4
EPOCHS = 2
MAX_LEN = 1024

# ===========================
# Load Dataset
# ===========================
# The file should have one JSON per line like {"text": "<bos> ... <eos>"}
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_data, val_data = dataset["train"], dataset["test"]

print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")

# ===========================
# Load Tokenizer & Model
# ===========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,  # ✅ 4-bit QLoRA setup
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Prepare model for QLoRA fine-tuning
model = prepare_model_for_kbit_training(model)

# ===========================
# LoRA Configuration
# ===========================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # For LLaMA family
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===========================
# Tokenization Function
# ===========================
def tokenize_function(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    result["labels"] = result["input_ids"].clone()
    return result

train_tokenized = train_data.map(tokenize_function, batched=True, remove_columns=["text"])
val_tokenized = val_data.map(tokenize_function, batched=True, remove_columns=["text"])

# ===========================
# Data Collator
# ===========================
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ===========================
# Training Arguments
# ===========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=EPOCHS,
    warmup_ratio=0.03,
    logging_steps=25,
    fp16=True,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    save_total_limit=3,
    report_to="none",  # Disable wandb for Colab/Kaggle
)

# ===========================
# Trainer
# ===========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
)

# ===========================
# Train the Model
# ===========================
trainer.train()

# ===========================
# Save Everything for Phase 2 & Deployment
# ===========================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)

# Save PEFT adapter weights (lightweight)
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))

print("✅ Phase 1 training complete! Model and LoRA adapter saved to:", OUTPUT_DIR)
