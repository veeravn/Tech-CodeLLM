from azure.ai.ml import MLClient
from azure.ai.ml.entities import CommandJob
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
import argparse
import os

# Workspace config
SUBSCRIPTION_ID = "7cc0da73-46c1-4826-a73b-d7e49a39d6c1"
RESOURCE_GROUP = "custom-tech-llm"
WORKSPACE_NAME = "Tech-LLM"
COMPUTE_NAME = "codegen-cluster"

def trigger_training(data_path):
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )

    # Define environment (or reuse existing one)
    env = Environment(
        name="codellama-env",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",  # base image
        conda_file="conda.yaml"  # You can define this file separately
    )

    # Define the fine-tuning job
    job = CommandJob(
        code="./",  # source directory
        command="python azure_ml_finetune.py --data_path ${{inputs.data_path}}",
        inputs={"data_path": data_path},
        environment=env,
        compute=COMPUTE_NAME,
        experiment_name="CodeLLaMA-Finetuning"
    )

    # Submit job
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job submitted. Track it in Azure ML Studio: {returned_job.studio_url}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to the new dataset JSONL file")
    args = parser.parse_args()
    trigger_training(args.data_path)
