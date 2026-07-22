# ============================================================
# migrate_add_contact_name.py - one-time schema migration
# ============================================================
# Porter-provided leads may identify a specific person to call.
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
    ALTER TABLE lead_candidates
    ADD COLUMN IF NOT EXISTS contact_name text NULL
""")

conn.commit()
print("Migration applied: lead_candidates.contact_name (text, nullable) ensured.")

cursor.execute("""
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_name = 'lead_candidates' AND column_name = 'contact_name'
""")
print(f"Verified: {cursor.fetchone()}")

cursor.close()
conn.close()
