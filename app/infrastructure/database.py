"""Async PostgreSQL repository for lead and call outcomes."""

from __future__ import annotations

import os
from typing import Any
from uuid import UUID

import asyncpg


def _lead_id(value: str | UUID) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


class PostgresRepository:
    def __init__(self, pool: asyncpg.Pool, recontact_days: int = 7) -> None:
        self._pool = pool
        self._recontact_days = recontact_days

    @classmethod
    async def connect(cls) -> "PostgresRepository":
        dsn = os.getenv("DATABASE_URL")
        options: dict[str, Any] = {
            "min_size": int(os.getenv("DB_POOL_MIN_SIZE", "1")),
            "max_size": int(os.getenv("DB_POOL_MAX_SIZE", "4")),
            "command_timeout": float(os.getenv("DB_COMMAND_TIMEOUT", "10")),
        }
        if dsn:
            pool = await asyncpg.create_pool(dsn=dsn, **options)
        else:
            pool = await asyncpg.create_pool(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "5432")),
                database=os.getenv("DB_NAME", "porter_leads"),
                user=os.getenv("DB_USER", "porter"),
                password=os.getenv("DB_PASSWORD", ""),
                **options,
            )
        return cls(pool, int(os.getenv("RECONTACT_DEFAULT_DAYS", "7")))

    async def close(self) -> None:
        await self._pool.close()

    async def get_next_lead(self) -> dict[str, Any] | None:
        row = await self._pool.fetchrow(
            """
            SELECT
                c.canonical_name AS company_name,
                c.city,
                c.state,
                c.industry,
                c.naics_code,
                c.website_domain,
                cc.phone,
                lc.id AS lead_candidate_id,
                lc.tier,
                lc.current_score,
                lc.why_now_summary,
                lc.contact_name
            FROM lead_candidates lc
            JOIN companies c ON c.id = lc.company_id
            LEFT JOIN LATERAL (
                SELECT candidate.phone
                FROM company_contactability candidate
                WHERE candidate.lead_candidate_id = lc.id
                  AND candidate.contactability_status NOT IN (
                      'invalid', 'disconnected'
                  )
                ORDER BY candidate.created_at DESC
                LIMIT 1
            ) cc ON true
            WHERE lc.status = 'active'
              AND lc.sector_excluded = false
              AND lc.sales_status = 'research'
              AND NOT EXISTS (
                  SELECT 1
                  FROM suppression_list sl
                  WHERE sl.active = true
                    AND (
                        (sl.match_type = 'company_name' AND sl.match_value = c.canonical_name)
                        OR (sl.match_type = 'domain' AND sl.match_value = c.website_domain)
                    )
              )
            ORDER BY
                CASE lc.tier
                    WHEN 'Hot' THEN 1
                    WHEN 'Warm' THEN 2
                    WHEN 'Cold' THEN 3
                    ELSE 4
                END,
                lc.current_score DESC
            LIMIT 1
            """
        )
        if row is None:
            return None
        lead = dict(row)
        lead["lead_candidate_id"] = str(lead["lead_candidate_id"])
        return lead

    async def create_call(
        self,
        lead_candidate_id: str | UUID,
        agent_version: str,
        llm_provider: str,
        tts_provider: str,
        room_id: str,
    ) -> int:
        return await self._pool.fetchval(
            """
            INSERT INTO calls (
                lead_candidate_id, call_started_at, agent_version,
                llm_provider, tts_provider, room_id
            )
            VALUES ($1, now(), $2, $3, $4, $5)
            RETURNING id
            """,
            _lead_id(lead_candidate_id),
            agent_version,
            llm_provider,
            tts_provider,
            room_id,
        )

    async def complete_call(
        self,
        call_id: int,
        lead_candidate_id: str | UUID,
        result: str,
        *,
        referral_details: str | None = None,
        callback_details: str | None = None,
        objection_text: str | None = None,
        update_lead: bool = True,
    ) -> None:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    UPDATE calls
                    SET call_ended_at = now(), final_call_result = $1
                    WHERE id = $2
                    """,
                    result,
                    call_id,
                )
                if update_lead:
                    await self._update_lead(
                        connection,
                        lead_candidate_id,
                        result,
                        referral_details,
                        callback_details,
                        objection_text,
                    )

    async def _update_lead(
        self,
        connection: asyncpg.Connection,
        lead_candidate_id: str | UUID,
        status: str,
        referral_details: str | None,
        callback_details: str | None,
        objection_text: str | None,
    ) -> None:
        await connection.execute(
            """
            UPDATE lead_candidates
            SET sales_status = $1,
                updated_at = now(),
                recontact_at = CASE
                    WHEN $1 = 'callback_later'
                    THEN now() + ($2 * interval '1 day')
                    ELSE recontact_at
                END,
                referral_details = COALESCE($3, referral_details),
                callback_details = COALESCE($4, callback_details),
                objection_text = COALESCE($5, objection_text)
            WHERE id = $6
            """,
            status,
            self._recontact_days,
            referral_details,
            callback_details,
            objection_text,
            _lead_id(lead_candidate_id),
        )

    async def add_to_suppression(
        self,
        company_name: str,
        website_domain: str,
        reason: str = "opted_out",
    ) -> None:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                if company_name:
                    await connection.execute(
                        """
                        INSERT INTO suppression_list
                            (match_type, match_value, reason, source)
                        VALUES ('company_name', $1, $2, 'voice_agent')
                        ON CONFLICT DO NOTHING
                        """,
                        company_name,
                        reason,
                    )
                if website_domain:
                    await connection.execute(
                        """
                        INSERT INTO suppression_list
                            (match_type, match_value, reason, source)
                        VALUES ('domain', $1, $2, 'voice_agent')
                        ON CONFLICT DO NOTHING
                        """,
                        website_domain,
                        reason,
                    )
