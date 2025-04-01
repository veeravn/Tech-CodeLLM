from fastapi import APIRouter
from pathlib import Path
import json

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/chat-history")
def chat_history():
    path = Path("chat_history.jsonl")
    if not path.exists():
        return {"history": []}

    try:
        with open(path, "r", encoding="utf-8") as f:
            history = [json.loads(line) for line in f if line.strip()]
        return {"history": history}
    except Exception as e:
        return {"error": str(e)}

@router.get("/version")
def version():
    return {"model": "CodeLLaMA-13B-Finetuned"}