"""
Model loader singleton for fine-tuned LLaMA model
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
from dotenv import load_dotenv

load_dotenv()


class ModelWrapper:
    """
    Singleton wrapper for the fine-tuned LLaMA model
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelWrapper, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not ModelWrapper._initialized:
            self.model_path = os.getenv("MERGED_MODEL_PATH", "/Users/aniketpatel/Desktop/CS460/final_model")
            print(f"🔹 Loading merged model from: {self.model_path}")
            
            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Check if model already has quantization config
            import json
            config_path = os.path.join(self.model_path, "config.json")
            has_quant_config = False
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    has_quant_config = "quantization_config" in config
            
            if has_quant_config:
                # Model is already quantized, load without additional quantization
                print("📦 Model already quantized, loading as-is...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            else:
                # Apply 4-bit quantization config
                print("🔧 Applying 4-bit quantization...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            
            self.model.eval()
            
            ModelWrapper._initialized = True
            print("✅ Model loaded successfully")

    def generate_from_prompt(self, prompt: str, max_new_tokens: int = 200, **gen_kwargs) -> str:
        """
        Generate response from the model given a complete prompt
        
        Args:
            prompt: The complete formatted prompt
            max_new_tokens: Maximum number of tokens to generate
            **gen_kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=gen_kwargs.get("temperature", 0.4),
                top_p=gen_kwargs.get("top_p", 0.85),
                repetition_penalty=gen_kwargs.get("repetition_penalty", 1.2),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded


# Global singleton instance
model_wrapper = None


def get_model_wrapper() -> ModelWrapper:
    """
    Get the global model wrapper instance
    """
    global model_wrapper
    if model_wrapper is None:
        model_wrapper = ModelWrapper()
    return model_wrapper

