import os
from dotenv import load_dotenv

# Optional: Load .env file for local dev
load_dotenv()

# Model & Inference Settings
USE_REMOTE_MODEL = os.getenv("USE_REMOTE_MODEL", "true").lower() == "true"
AML_ENDPOINT_URL = os.getenv("AML_ENDPOINT_URL")
AML_API_KEY = os.getenv("AML_API_KEY")
LOCAL_MODEL_PATH = os.getenv("MODEL_PATH", "./output_finetuned_latest")

# Security
JWT_SECRET = os.getenv("JWT_SECRET", "changeme")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# File paths
CHAT_HISTORY_PATH = os.getenv("CHAT_HISTORY_PATH", "chat_history.jsonl")

# Azure ML Context (optional)
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
AZURE_WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME")