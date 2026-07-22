"""Apply immutable PostgreSQL migrations in filename order."""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import asyncpg

from scripts.db_utils import connect


MIGRATIONS_DIR = Path(__file__).parents[1] / "migrations"
MIGRATION_NAME = re.compile(r"^\d{4}_[a-z0-9_]+\.sql$")
HISTORY_TABLE = "public.schema_migrations"


@dataclass(frozen=True, slots=True)
class Migration:
    filename: str
    sql: str
    checksum: str


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Migration directory not found: {directory}")

    paths = sorted(directory.glob("*.sql"), key=lambda path: path.name)
    if not paths:
        raise RuntimeError(f"No SQL migrations found in {directory}")

    migrations: list[Migration] = []
    prefixes: set[str] = set()
    for path in paths:
        if not MIGRATION_NAME.fullmatch(path.name):
            raise ValueError(f"Invalid migration filename: {path.name}")
        prefix = path.name[:4]
        if prefix in prefixes:
            raise ValueError(f"Duplicate migration sequence: {prefix}")
        prefixes.add(prefix)
        content = path.read_bytes()
        migrations.append(
            Migration(
                filename=path.name,
                sql=content.decode("utf-8"),
                checksum=hashlib.sha256(content).hexdigest(),
            )
        )
    return migrations


async def _history_exists(connection: asyncpg.Connection) -> bool:
    return (
        await connection.fetchval("SELECT to_regclass($1)", HISTORY_TABLE)
        is not None
    )


async def _applied_checksum(
    connection: asyncpg.Connection, filename: str
) -> str | None:
    return await connection.fetchval(
        "SELECT checksum FROM schema_migrations WHERE filename = $1",
        filename,
    )


async def _apply_migration(
    connection: asyncpg.Connection, migration: Migration
) -> None:
    async with connection.transaction():
        await connection.execute(migration.sql)
        await connection.execute(
            """
            INSERT INTO schema_migrations (filename, checksum)
            VALUES ($1, $2)
            """,
            migration.filename,
            migration.checksum,
        )


async def run_migrations(
    connection: asyncpg.Connection,
    migrations: list[Migration],
) -> list[str]:
    applied: list[str] = []
    history_exists = await _history_exists(connection)

    for position, migration in enumerate(migrations):
        if not history_exists:
            if position != 0:
                raise RuntimeError(
                    "The first migration must create public.schema_migrations"
                )
            await _apply_migration(connection, migration)
            history_exists = True
            applied.append(migration.filename)
            continue

        existing_checksum = await _applied_checksum(connection, migration.filename)
        if existing_checksum is not None:
            if existing_checksum != migration.checksum:
                raise RuntimeError(
                    f"Applied migration checksum changed: {migration.filename}"
                )
            continue

        await _apply_migration(connection, migration)
        applied.append(migration.filename)

    return applied


async def main() -> None:
    migrations = discover_migrations()
    connection = await connect()
    try:
        applied = await run_migrations(connection, migrations)
    finally:
        await connection.close()

    if applied:
        for filename in applied:
            print(f"Applied: {filename}")
    else:
        print("Database schema is up to date.")


if __name__ == "__main__":
    asyncio.run(main())
