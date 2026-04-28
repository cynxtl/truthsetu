"""
TruthSetu — FastAPI Application
"""
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
    # ── Startup ───────────────────────────────────────────────
    logger.info("TruthSetu starting up...")
    await connect_db()

    # Start background scheduler
    from backend.core.scheduler import start_scheduler
    await start_scheduler()

    # Run initial RSS fetch on startup
    try:
        from backend.core.rss_monitor import get_rss_monitor
        monitor = await get_rss_monitor()
        logger.info("Running initial RSS fetch...")
        await monitor.refresh_faiss_index()
        await monitor.monitor_fact_checkers()
        logger.success("Initial RSS fetch complete ✓")
    except Exception as e:
        logger.warning(f"Initial RSS fetch failed (non-fatal): {e}")

    yield

    # ── Shutdown ──────────────────────────────────────────────
    from backend.core.scheduler import stop_scheduler
    await stop_scheduler()
    await close_db()
    logger.info("TruthSetu shutdown complete.")


app = FastAPI(
    title="TruthSetu API",
    description="AI-Based Multi-Agent System to Detect, Verify "
                "and Counter Crisis Misinformation in Real Time",
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


# scheduler_status moved to routes.py