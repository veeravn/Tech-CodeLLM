import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = None
tokenizer = None

def init():
    global base_model, tokenizer

    # Get model directory injected by Azure ML
    model_dir = os.path.join(os.getenv("AZUREML_MODEL_DIR", ""), "model_output")
    offload_dir = "/tmp/offload"
    os.makedirs("/tmp/offload", exist_ok=True)

    # Load the base model (supports LoRA adapters + float16 if GPU available)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        offload_folder="/tmp/offload"
    )
    base_model.config.use_cache = True

    # Load tokenizer and fix padding token
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"✅ Model and tokenizer loaded from: {model_dir}")


def run(raw_data):
    global base_model, tokenizer

    try:
        if isinstance(raw_data, str):
            inputs = raw_data
        elif isinstance(raw_data, dict) and "inputs" in raw_data:
            inputs = raw_data["inputs"]
        else:
            return {"error": "Input format not recognized"}

        # Tokenize input
        input_tokens = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(base_model.device)

        # Generate output
        with torch.no_grad():
            output_tokens = base_model.generate(**input_tokens, max_new_tokens=100)

        # Decode and return
        result = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return {"output": result}

    except Exception as e:
        return {"error": str(e)}
