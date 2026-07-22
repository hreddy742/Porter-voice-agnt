"""Shared asyncpg connection helper for administrative scripts."""

import os

import asyncpg
from dotenv import load_dotenv


async def connect() -> asyncpg.Connection:
    load_dotenv()
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return await asyncpg.connect(dsn)
    return await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "porter_leads"),
        user=os.getenv("DB_USER", "porter"),
        password=os.getenv("DB_PASSWORD", ""),
    )
