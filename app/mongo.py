"""
MongoDB Atlas Client
====================
Shared async MongoDB connection for OptionsGangster.
Uses motor (async driver) for non-blocking access.

Collections:
  trade_log     — every alert resolved with outcome + R-multiple
  daily_session — daily capital-mode state (loss streaks, locks)
"""
from __future__ import annotations

import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings

logger = logging.getLogger("optionsgangster")

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


def get_db() -> AsyncIOMotorDatabase:
    """
    Return the shared AsyncIOMotorDatabase instance.
    Lazily creates the client on first call.
    """
    global _client, _db
    if _db is None:
        uri = settings.MONGO_URI
        db_name = settings.MONGO_DB
        if not uri:
            raise RuntimeError("MONGO_URI is not configured")
        _client = AsyncIOMotorClient(uri)
        _db = _client[db_name]
        logger.info(f"[MongoDB] Connected to {db_name}")
    return _db


async def close_db():
    """Close the motor client (call on app shutdown)."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("[MongoDB] Connection closed")
