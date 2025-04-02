from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model as MLModel
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.core.exceptions import HttpResponseError
from azure.identity import ManagedIdentityCredential


MODEL_NAME = "codellama/CodeLlama-13b-hf"

# Authenticate with Azure ML using Managed Identity
def main(data_path):
    print("🚀 Starting fine-tuning script")
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    credential = ManagedIdentityCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_WORKSPACE_NAME"]
    )
    print("Connected to Azure ML Workspace:", ml_client.workspace_name)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["instruction"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        model_inputs["labels"] = model_inputs["input_ids"]
        return model_inputs
    # Load dataset
    dataset = load_dataset("json", data_files=data_path)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # LoRA config
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

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,               # stays fixed for easier resume
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=True,
        logging_dir="./logs",
        save_strategy="steps",                # ✅ checkpoint every N steps
        save_steps=500,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=50,
        push_to_hub=False,
        remove_unused_columns=False           # ✅ needed for PEFT
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset = tokenized_datasets["train"].select(range(100)),
    )
    # Train model
    print("🚀 Starting training (auto-resume enabled)...")
    print(f"OutputDir: {OUTPUT_DIR}")
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

    print(f"Last Checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint if last_checkpoint else None)

    # Save model and tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model fine-tuned and saved at {OUTPUT_DIR}")

    # Register model
    model_info = ml_client.models.create_or_update(
        MLModel(
            name="TechLLM-Model",
            path=OUTPUT_DIR,
            description="Fine-tuned CodeLLaMA 13B on tech dataset",
            tags={"timestamp": timestamp, "source": os.path.basename(data_path)}
        )
    )
    print(f"✅ Model registered: {model_info.name} v{model_info.version}")
    deploy_latest_model(ml_client, model_info.name, model_info.version)

def deploy_latest_model(ml_client, model_name: str, model_version: str):
    endpoint_name = "techllm-endpoint"
    deployment_name = "blue"

    # Check if endpoint exists
    try:
        ml_client.online_endpoints.get(name=endpoint_name)
        print(f"✅ Endpoint '{endpoint_name}' exists.")
    except HttpResponseError as e:
        # Create it
        if "NotFound" in str(e):
            print(f"Endpoint '{endpoint_name}' not found, creating it.")
            # create endpoint
            endpoint = ManagedOnlineEndpoint(
                name=endpoint_name,
                auth_mode="key"
            )
        else:
            raise
        
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"🆕 Created new endpoint '{endpoint_name}'.")

    # Create or update deployment
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=f"{model_name}:{model_version}",
        instance_type="Standard_NC6",  # or Standard_DS3_v2 if CPU
        instance_count=1,
        environment_variables={"TRANSFORMERS_CACHE": "/tmp"},
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    # Set traffic to 100%
    ml_client.online_endpoints.begin_update(
        name=endpoint_name,
        traffic={"blue": 100}
    ).result()

    print(f"🚀 Deployed model {model_name}:{model_version} to '{endpoint_name}' with 100% traffic.")


if __name__ == "__main__":
    # Parse arguments for dataset path
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save model outputs")
    args = parser.parse_args()

    DATA_PATH = args.data_path
    OUTPUT_DIR = args.output_dir
    print(f"[INFO] Output directory set to: {OUTPUT_DIR}")
    print("[DEBUG] Files in output directory before training:")
    print(os.listdir(OUTPUT_DIR))
    main(DATA_PATH)