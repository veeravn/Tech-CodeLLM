from azure.ai.ml import MLClient, Input, command
from azure.identity import DefaultAzureCredential
import argparse

# Authenticate and connect to Azure ML
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id="7cc0da73-46c1-4826-a73b-d7e49a39d6c1",
    resource_group_name="custom-tech-llm",
    workspace_name="Tech-LLM"
)

def submit_finetune_job(data_path):
    # Define training job
    job = command(
        code="./",
        command="python azure_ml_finetune.py --data_path ${{inputs.training_data}}",
        inputs={
            "training_data": Input(
                type="uri_file",
                mode="ro_mount",
                path=data_path
            )
        },
        environment="code-llama-env@latest",
        compute="codegen-cluster",
        display_name="Finetune-CodeLLaMA",
        description="LoRA fine-tuning of CodeLLaMA-13B with tech instruction dataset",
        environment_variables={
            "AZURE_SUBSCRIPTION_ID": "7cc0da73-46c1-4826-a73b-d7e49a39d6c1",
            "AZURE_RESOURCE_GROUP": "custom-tech-llm",
            "AZURE_WORKSPACE_NAME": "Tech-LLM"
        }
    )

    # Submit job
    returnedJob = ml_client.jobs.create_or_update(job, experiment_name="codellama")
    print("✅ Job submitted to Azure ML.")
    return returnedJob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Full path to the dataset (URI or SAS URL)")
    args = parser.parse_args()

    submit_finetune_job(args.data_path)
