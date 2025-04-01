from azure_ml_env_setup import submit_finetune_job
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import os

# Workspace config
SUBSCRIPTION_ID = "7cc0da73-46c1-4826-a73b-d7e49a39d6c1"
RESOURCE_GROUP = "custom-tech-llm"
WORKSPACE_NAME = "Tech-LLM"
COMPUTE_NAME = "codegen-cluster"

credential = DefaultAzureCredential()

# Load latest file from blob (assumes naming like new_data_YYYYMMDDHHMM.jsonl)
def get_latest_data_path():
    blob_service_client = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])
    container_name = os.environ["CONTAINER_NAME"]
    container_client = blob_service_client.get_container_client(container_name)

    blobs = list(container_client.list_blobs(name_starts_with="new_data_"))
    if not blobs:
        raise RuntimeError("❌ No new_data_ files found in blob container.")

    latest_blob = sorted(blobs, key=lambda b: b.name, reverse=True)[0]
    url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{latest_blob.name}"
    print(f"🧾 Using latest dataset: {latest_blob.name}")
    return url

def trigger_training():
    latest_data_url = get_latest_data_path()
    submit_finetune_job(data_path=latest_data_url)
