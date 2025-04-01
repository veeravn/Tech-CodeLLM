from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..services.logger import append_chat_entry

router = APIRouter()

class ChatLog(BaseModel):
    instruction: str
    response: str
    input: str = ""

@router.post("/log-chat")
def log_chat(payload: ChatLog):
    if not payload.instruction or not payload.response:
        raise HTTPException(status_code=400, detail="instruction and response required")

    success = append_chat_entry(payload.instruction, payload.response, payload.input)
    return {"status": "ok" if success else "duplicate"}