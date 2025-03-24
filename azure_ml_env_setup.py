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

# Get dataset reference
#dataset_input = ml_client.data.get(name="instruction_dataset", version="1")
# Define training job
job = command(
    code="./",
    command="python azure_ml_finetune.py --data_path ${{inputs.training_data}}",
    inputs={
        "training_data": Input(type="uri_file", path="https://techllm4159171971.blob.core.windows.net/tech-instruction-dataset/instruction_dataset.jsonl")
    },
    environment="codellama-env@latest",  # Change if you're using a custom environment
    compute="codegen-cluster",
    display_name="Finetune-CodeLLaMA",
    description="LoRA fine-tuning of CodeLLaMA-13B with tech instruction dataset"
)

# Submit job
ml_client.jobs.create_or_update(job)
print("Job submitted to Azure ML.")
