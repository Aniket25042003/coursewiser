from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# ---------------------------
# 1. Load merged model
# ---------------------------
MERGED_MODEL_PATH = "./merged_model"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("🔹 Loading merged model...")
tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()

# ---------------------------
# 2. Inference function
# ---------------------------
def generate_response(prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------
# 3. Test the model
# ---------------------------
prompt = "Explain the different data structures."
response = generate_response(prompt)
print("\n🧠 Model Response:\n", response)
