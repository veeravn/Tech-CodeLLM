import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model as MLModel
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.ai.ml.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential

# Parse arguments for dataset path
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

OUTPUT_DIR = "./output_finetuned_model"
MODEL_NAME = "codellama/CodeLlama-13b-hf"

# Authenticate with Azure ML using Managed Identity
def main(data_path):
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_WORKSPACE_NAME"]
    )
    print("Connected to Azure ML Workspace:", ml_client.workspace_name)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["instruction"], truncation=True, padding="max_length", max_length=512)

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
        output_dir=OUTPUT_DIR,
        evaluation_strategy="no",
        save_strategy="epoch",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=2e-5,
        fp16=True,
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["train"][:100],
    )

    # Train model
    print("Starting fine-tuning...")
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model fine-tuned and saved at {OUTPUT_DIR}")

    # Register model
    ml_client.models.create_or_update(
        MLModel(
            name="CodeLLaMA_13B_Finetuned",
            path=OUTPUT_DIR,
            description="LoRA-finetuned CodeLLaMA 13B model on tech instructions"
        )
    )
    print("Model registered in Azure ML.")

    model_info = ml_client.models.create_or_update(...)
    print(f"✅ Model registered: {model_info.name} v{model_info.version}")
    deploy_latest_model(ml_client, model_info.name, model_info.version)

def deploy_latest_model(ml_client, model_name: str, model_version: str):
    endpoint_name = "techllm-endpoint"
    deployment_name = "blue"

    # Check if endpoint exists
    try:
        ml_client.online_endpoints.get(name=endpoint_name)
        print(f"✅ Endpoint '{endpoint_name}' exists.")
    except ResourceNotFoundError:
        # Create it
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            auth_mode="key"
        )
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to dataset (.jsonl)")
    args = parser.parse_args()
    main(args.data_path)