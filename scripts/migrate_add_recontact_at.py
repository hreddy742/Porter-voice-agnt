# ============================================================
# migrate_add_recontact_at.py — one-time schema migration
# ============================================================
# Adds lead_candidates.recontact_at (timestamptz, nullable).
#
# Written when 'callback_later' (bad-timing decline) was added to
# the call flow — that result needs somewhere to record "come back
# to this lead after <date>", and no such column existed (confirmed
# against the live schema: only created_at/updated_at/deleted_at).
#
# Safe to run more than once: IF NOT EXISTS makes it a no-op if the
# column is already there.
# ============================================================

from app.db import get_connection

conn = get_connection()
cursor = conn.cursor()

cursor.execute("""
    ALTER TABLE lead_candidates
    ADD COLUMN IF NOT EXISTS recontact_at timestamptz NULL
""")

conn.commit()
print("Migration applied: lead_candidates.recontact_at (timestamptz, nullable) ensured.")

cursor.execute("""
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_name = 'lead_candidates' AND column_name = 'recontact_at'
""")
row = cursor.fetchone()
print(f"Verified: {row}")

cursor.close()
conn.close()
