from azure.ai.ml import MLClient, Input, Output, command
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
compute_target_name = "codegen-cluster"

# Reuse existing environment
env = ml_client.environments.get(name="code-llama-env", label="latest")

def submit_finetune_job(data_path):

    # Input: dataset in AzureML URI format
    train_data = Input(
        type="uri_file",
        path=f"azureml://subscriptions/7cc0da73-46c1-4826-a73b-d7e49a39d6c1/resourcegroups/custom-tech-llm/workspaces/Tech-LLM/datastores/instructions/paths/{data_path}",
        mode="ro_mount"
    )

    # Define training job
    job = command(
        code="./",
        command="python azure_ml_finetune.py --data_path ${{inputs.training_data}}",
        inputs={"training_data": train_data},
        outputs={"output_dir": 
                 Output(path="azureml://datastores/workspaceblobstore/paths/codellama-outputs",
                        type="uri_folder",
                        mode="rw_mount")
                },
        environment=env,
        compute=compute_target_name,
        experiment_name="codellama",
        display_name="codellama-ddp-job",
        description="LoRA fine-tuning of CodeLLaMA-13B with tech instruction dataset",
        environment_variables={
            "AZURE_SUBSCRIPTION_ID": "7cc0da73-46c1-4826-a73b-d7e49a39d6c1",
            "AZURE_RESOURCE_GROUP": "custom-tech-llm",
            "AZURE_WORKSPACE_NAME": "Tech-LLM"
        }
    )

    # Submit to experiment
    returnedJob = ml_client.jobs.create_or_update(job)
    print("✅ Job submitted to Azure ML.")
    return returnedJob
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Name of the dataset file.  File must be located in the tech-instruction-dataset container.")
    args = parser.parse_args()

    submit_finetune_job(args.data_path)
