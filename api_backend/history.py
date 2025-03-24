from fastapi import APIRouter
import json
from rotate_chat_history import rotate_chat_history

router = APIRouter()

@router.get("/chat-history")
def read_history():
    try:
        with open("chat_history.jsonl", "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f.readlines()]
    except Exception as e:
        return {"error": str(e)}

@router.post("/rotate-history")
def rotate():
    rotate_chat_history()
    return {"status": "chat history rotated"}
