from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def run(data):
    prompt = data.get("prompt", "")
    result = pipe(prompt, max_new_tokens=200)[0]["generated_text"]
    return {"output": result}
