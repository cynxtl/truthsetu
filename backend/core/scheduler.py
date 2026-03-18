"""
TruthSetu — APScheduler
========================
Background jobs that run automatically:

  Every 15 min → monitor fact-checker RSS feeds → seed LEARN
  Every 30 min → refresh all RSS feeds → update FAISS
  Every 60 min → rebuild full FAISS index from scratch
  Every 24 hrs → retrain ML classifier with new examples
"""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
from typing import Optional

_scheduler: Optional[AsyncIOScheduler] = None


async def start_scheduler():
    """Start all background jobs. Called on server startup."""
    global _scheduler

    _scheduler = AsyncIOScheduler()

    # ── Job 1: Monitor fact-checkers every 15 min ─────────────
    _scheduler.add_job(
        _monitor_fact_checkers,
        trigger=IntervalTrigger(minutes=15),
        id="monitor_fact_checkers",
        name="Monitor Alt News + BOOM for new debunks",
        replace_existing=True,
        misfire_grace_time=60,
    )

    # ── Job 2: Refresh RSS → update FAISS every 30 min ────────
    _scheduler.add_job(
        _refresh_rss_feeds,
        trigger=IntervalTrigger(minutes=30),
        id="refresh_rss_feeds",
        name="Refresh RSS feeds and update FAISS",
        replace_existing=True,
        misfire_grace_time=120,
    )

    # ── Job 3: Full FAISS rebuild every 6 hours ───────────────
    _scheduler.add_job(
        _rebuild_full_index,
        trigger=IntervalTrigger(hours=6),
        id="rebuild_full_index",
        name="Full FAISS index rebuild",
        replace_existing=True,
        misfire_grace_time=300,
    )

    _scheduler.start()
    logger.success("Scheduler started ✓")
    logger.info("  → Fact-checker monitor: every 15 min")
    logger.info("  → RSS feed refresh:     every 30 min")
    logger.info("  → Full index rebuild:   every 6 hours")


async def stop_scheduler():
    """Stop scheduler on server shutdown."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown()
        logger.info("Scheduler stopped.")


# ── Job functions ─────────────────────────────────────────────

async def _monitor_fact_checkers():
    """Monitor Alt News + BOOM + Vishvas for new debunks."""
    try:
        from backend.core.rss_monitor import get_rss_monitor
        monitor = await get_rss_monitor()
        await monitor.monitor_fact_checkers()
    except Exception as e:
        logger.error(f"Fact-checker monitor job failed: {e}")


async def _refresh_rss_feeds():
    """Fetch all RSS sources and update FAISS with new entries."""
    try:
        from backend.core.rss_monitor import get_rss_monitor
        monitor = await get_rss_monitor()
        await monitor.refresh_faiss_index()
    except Exception as e:
        logger.error(f"RSS refresh job failed: {e}")


async def _rebuild_full_index():
    """Full FAISS index rebuild from scratch every 6 hours."""
    try:
        from backend.agents.verify_agent import get_verify_agent
        agent = await get_verify_agent()
        await agent.rebuild_index()
        logger.success("Full FAISS index rebuilt ✓")
    except Exception as e:
        logger.error(f"Full index rebuild failed: {e}")