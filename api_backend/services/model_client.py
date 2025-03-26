import os
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

USE_REMOTE_ENDPOINT = bool(os.environ.get("AML_ENDPOINT_URL"))

if not USE_REMOTE_ENDPOINT:
    model_path = os.environ.get("MODEL_PATH", "./output_finetuned_latest")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)


def generate_response(prompt: str):
    if USE_REMOTE_ENDPOINT:
        aml_url = os.environ["AML_ENDPOINT_URL"]
        headers = {
            "Authorization": f"Bearer {os.environ['AML_API_KEY']}",
            "Content-Type": "application/json"
        }
        res = requests.post(aml_url, headers=headers, json={"prompt": prompt})
        if res.status_code != 200:
            raise RuntimeError(f"Inference failed: {res.text}")
        return res.json().get("output", res.json())
    else:
        response = pipe(prompt, max_new_tokens=200, do_sample=True, top_p=0.95, temperature=0.7)
        return response[0]["generated_text"]