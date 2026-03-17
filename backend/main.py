from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from backend.db.mongodb import connect_db, close_db
from backend.api.routes import router
from backend.core.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TruthSetu starting up...")
    await connect_db()
    yield
    logger.info("TruthSetu shutting down...")
    await close_db()


app = FastAPI(
    title="TruthSetu API",
    description="AI-Based Multi-Agent System to Detect, Verify and Counter Crisis Misinformation",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "app":    settings.app_name,
        "env":    settings.app_env,
        "llm":    settings.llm_provider,
    }