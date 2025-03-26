from fastapi import FastAPI
from routes.inference import router as inference_router
from routes.log import router as log_router
from routes.health import router as health_router

app = FastAPI(title="Custom Tech LLM API")

# Register route modules
app.include_router(inference_router)
app.include_router(log_router)
app.include_router(health_router)