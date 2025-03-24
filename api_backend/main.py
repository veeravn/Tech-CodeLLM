from fastapi import FastAPI
from api_backend.chat_logger import router as chat_logger_router
from api_backend.inference import router as inference_router
from api_backend.history import router as history_router

app = FastAPI(title="Custom Tech LLM API")

# Register endpoints
app.include_router(chat_logger_router)
app.include_router(inference_router)
app.include_router(history_router)
