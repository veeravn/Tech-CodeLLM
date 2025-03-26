from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from ..auth import require_jwt
from ..services.model_client import generate_response
from ..services.logger import append_chat_entry

router = APIRouter()

class InferenceRequest(BaseModel):
    prompt: str

@router.post("/inference")
def inference(payload: InferenceRequest, user=Depends(require_jwt)):
    if not payload.prompt:
        raise HTTPException(status_code=400, detail="Prompt required")
    result = generate_response(prompt=payload.prompt)
    # Log the prompt and output as a chat entry
    append_chat_entry(
        instruction=payload.prompt,
        response=result,
        input_text=""
    )

    return {"output": result}