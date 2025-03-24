from fastapi import APIRouter, HTTPException
from append_chat import append_chat_entry

router = APIRouter()

@router.post("/log-chat")
def log_chat(payload: dict):
    instruction = payload.get("instruction")
    response = payload.get("response")
    input_text = payload.get("input", "")

    if not instruction or not response:
        raise HTTPException(status_code=400, detail="instruction and response required")

    success = append_chat_entry(instruction, response, input_text)
    return {"status": "ok" if success else "duplicate"}
