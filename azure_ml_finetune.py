import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
from azureml.core import Workspace, Dataset, Experiment, ScriptRunConfig, Environment, Model

# Connect to Azure ML Workspace
ws = Workspace(subscription_id="7cc0da73-46c1-4826-a73b-d7e49a39d6c1",
               resource_group="custom-tech-llm",
               workspace_name="Tech-LLM")
print("Connected to Azure ML Workspace:", ws.name)

# Set Compute Target
compute_target = "codegen-cluster"

all_datasets = Dataset.get_all(ws)
print("Datasets in workspace:")
for d in all_datasets:
    print(f"- {d.name}")

# Load dataset from Azure ML
dataset = Dataset.get_by_name(ws, name="instruction_dataset", version="1")
dataset_path = dataset.download(target_path="./", overwrite=True)

# Define model and dataset
MODEL_NAME = "codellama/CodeLlama-13b-hf"
DATASET_PATH = "./instruction_dataset.jsonl"
OUTPUT_DIR = "./output_finetuned_model"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    if "instruction" in examples:
        return tokenizer(examples["instruction"], truncation=True, padding="max_length", max_length=1000)
    else:
        prompts = [
            "Explain the following code:\n\n" + row["content"]
            for row in examples["row"]
            if "content" in row
        ]
        return tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=1000
        )

# Load dataset
dataset = load_dataset("json", data_files=DATASET_PATH)
train_dataset = dataset["train"]

tokenized_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=train_dataset.column_names  # ✅ access split directly
)

# Configure LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_dir="./logs",
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["train"][:100],
)

# Train model
trainer.train()

# Save model to Azure ML Model Registry
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model fine-tuned and saved at {OUTPUT_DIR}")

Model.register(workspace=ws, model_path=OUTPUT_DIR, model_name="CodeLLaMA_13B_Finetuned")
print("Model registered in Azure ML.")

# Configure environment for Azure ML
env = Environment.from_conda_specification(
    name="code-llama-env",
    file_path="codellama_env.yml"
)

env.docker.enabled = True
env.python.conda_dependencies.add_pip_package("transformers")
env.python.conda_dependencies.add_pip_package("peft")
env.python.conda_dependencies.add_pip_package("datasets")

# Submit training job to Azure ML
script_config = ScriptRunConfig(
    source_directory=".",
    script="azure_ml_finetune.py",
    compute_target="codegen-cluster",
    environment=env
)

experiment = Experiment(workspace=ws, name="CodeLLaMA-Finetuning")
run = experiment.submit(script_config)
print("Job submitted. View logs in Azure ML Studio.")