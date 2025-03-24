from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml import command, Input

# Authenticate and connect to Azure ML
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id="7cc0da73-46c1-4826-a73b-d7e49a39d6c1",
    resource_group_name="custom-tech-llm",
    workspace_name="Tech-LLM"
)

# Define training job
job = command(
    code="./",
    command="python azure_ml_finetune.py --data_path ${{inputs.training_data}}",
    inputs={
        "training_data": Input(type="uri_file", mode="ro_mount", path="azureml://subscriptions/7cc0da73-46c1-4826-a73b-d7e49a39d6c1/resourcegroups/custom-tech-llm/workspaces/Tech-LLM/datastores/workspaceartifactstore/paths/instruction_dataset.jsonl")
    },
    environment="code-llama-env@latest",  # Change if you're using a custom environment
    compute="codegen-cluster",
    display_name="Finetune-CodeLLaMA",
    description="LoRA fine-tuning of CodeLLaMA-13B with tech instruction dataset",
    environment_variables={"AZURE_SUBSCRIPTION_ID":"7cc0da73-46c1-4826-a73b-d7e49a39d6c1",
                            "AZURE_RESOURCE_GROUP":"custom-tech-llm",
                            "AZURE_WORKSPACE_NAME":"Tech-LLM"}
)

# Submit job
ml_client.jobs.create_or_update(job, experiment_name="codellama")
print("Job submitted to Azure ML.")
