from fastapi import APIRouter, Depends, HTTPException
from api_backend.auth import require_jwt
import requests
import os

router = APIRouter()

# Load endpoint info from env vars
AML_ENDPOINT_URL = os.environ.get("AML_ENDPOINT_URL")  # e.g. https://techllm-endpoint.eastus.inference.ml.azure.com/score
AML_API_KEY = os.environ.get("AML_API_KEY")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AML_API_KEY}"
}

@router.post("/generate")
def generate(payload: dict, user=Depends(require_jwt)):
    prompt = payload.get("prompt", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    try:
        response = requests.post(AML_ENDPOINT_URL, headers=headers, json={"prompt": prompt})
        if response.status_code != 200:
            raise Exception(f"Azure ML returned {response.status_code}: {response.text}")

        result = response.json()
        return {"output": result.get("output", result)}  # depends on your scoring script
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
