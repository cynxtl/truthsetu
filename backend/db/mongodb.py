from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from loguru import logger
from backend.core.config import get_settings

settings = get_settings()

_client: AsyncIOMotorClient = None


async def connect_db():
    global _client
    logger.info("Connecting to MongoDB...")
    _client = AsyncIOMotorClient(settings.mongodb_uri)
    await _client.admin.command("ping")
    logger.success("MongoDB connected ✓")


async def close_db():
    global _client
    if _client:
        _client.close()
        logger.info("MongoDB disconnected.")


def get_db() -> AsyncIOMotorDatabase:
    return _client[settings.mongodb_db_name]