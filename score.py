import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List
import logging

# Load model and tokenizer at startup
def init():
    global model, tokenizer
    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # Check for PEFT adapter
    adapter_path = os.path.join(model_dir, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        print("No LoRA adapter found, using base model.")
        model = base_model

    model.eval()
    print("✅ Model and tokenizer loaded.")

# Generate prediction
def run(data):
    try:
        input_text = data.get("input") or data["data"]
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=50,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return {"output": decoded}
    except Exception as e:
        logging.exception("Error in scoring script")
        return {"error": str(e)}
