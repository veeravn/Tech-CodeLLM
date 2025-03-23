from azureml.core import Workspace, Environment, ScriptRunConfig, Experiment, ComputeTarget
from azureml.exceptions import ComputeTargetException

def submit_job_from_sdk():
    # Connect to workspace
    ws = Workspace(subscription_id="7cc0da73-46c1-4826-a73b-d7e49a39d6c1",
                   resource_group="custom-tech-llm",
                   workspace_name="Tech-LLM")
    print(f"✅ Connected to Azure ML Workspace: {ws.name}")

    # Ensure compute target exists
    compute_target_name = "codegen-cluster"
    try:
        compute_target = ComputeTarget(workspace=ws, name=compute_target_name)
    except ComputeTargetException:
        raise RuntimeError(f"Compute target '{compute_target_name}' not found in workspace.")

    # Register or get environment
    env_name = "codellama-env"
    try:
        env = Environment.get(workspace=ws, name=env_name)
        print(f"✅ Environment '{env_name}' found.")
    except Exception:
        print(f"⚠️ Environment '{env_name}' not found. Creating and registering it...")
        env = Environment.from_conda_specification(
            name="codellama-env",
            file_path="codellama_env.yml"  # Update this if your file is named differently
        )
        env.register(workspace=ws)
        print(f"✅ Environment '{env_name}' registered.")

    # Create or get experiment
    experiment_name = "CodeLLaMA-Finetuning"
    experiment = Experiment(workspace=ws, name=experiment_name)

    # Prepare the training job
    script_config = ScriptRunConfig(
        source_directory=".",
        script="azure_ml_finetune.py",
        compute_target=compute_target,
        environment=env
    )

    # Submit job
    run = experiment.submit(script_config)
    print(f"✅ Job submitted to cluster '{compute_target_name}' in experiment '{experiment_name}'")
    print("🔗 Monitor the job in Azure ML Studio.")

if __name__ == "__main__":
    submit_job_from_sdk()
