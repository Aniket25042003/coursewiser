from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# ---------------------------
# 1. Paths
# ---------------------------
BASE_MODEL = "meta-llama/Llama-3.2-3B"
ADAPTER_PATH = "./phase2"                       # folder with adapter_model.safetensors
MERGED_MODEL_SAVE_PATH = "./final_model"       # output path

# ---------------------------
# 2. Load tokenizer
# ---------------------------
print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# ---------------------------
# 3. Load base model in 4-bit (if limited VRAM)
# ---------------------------
print("🔹 Loading base model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

# ---------------------------
# 4. Load and merge LoRA adapter
# ---------------------------
print("🔹 Loading and merging LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
merged_model = model.merge_and_unload()

# ---------------------------
# 5. Save merged model
# ---------------------------
print("💾 Saving merged model to:", MERGED_MODEL_SAVE_PATH)
merged_model.save_pretrained(MERGED_MODEL_SAVE_PATH)
tokenizer.save_pretrained(MERGED_MODEL_SAVE_PATH)

print("✅ Merge complete! Merged model saved successfully.")
