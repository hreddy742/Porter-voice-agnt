# ============================================================
# migrate_add_calls_tables.py - one-time schema migration
# ============================================================
# Stores one row per call and the ordered user/assistant transcript turns.
# Safe to run more than once: IF NOT EXISTS makes repeat runs a no-op.
# ============================================================

import os

import psycopg2
from dotenv import load_dotenv


load_dotenv()

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="porter_leads",
    user="porter",
    password=os.getenv("DB_PASSWORD", ""),
)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS calls (
        id SERIAL PRIMARY KEY,
        lead_candidate_id UUID NOT NULL REFERENCES lead_candidates(id),
        call_started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        call_ended_at TIMESTAMPTZ,
        agent_version TEXT,
        llm_provider TEXT,
        tts_provider TEXT,
        final_call_result TEXT,
        room_id TEXT
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS call_turns (
        id SERIAL PRIMARY KEY,
        call_id INTEGER NOT NULL REFERENCES calls(id),
        turn_number INTEGER NOT NULL,
        role TEXT NOT NULL,
        text TEXT NOT NULL,
        task_or_stage TEXT,
        timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
    )
""")

conn.commit()
print("Migration applied: calls and call_turns tables ensured.")

cursor.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_name IN ('calls', 'call_turns')
    ORDER BY table_name
""")
for row in cursor.fetchall():
    print(f"Verified: {row[0]}")

cursor.close()
conn.close()
