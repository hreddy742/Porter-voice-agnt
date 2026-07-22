# ============================================================
# migrate_add_call_notes_columns.py — one-time schema migration
# ============================================================
# Adds lead_candidates.referral_details, .callback_details, and
# .objection_text (all text, nullable).
#
# Written because none of these three were persisted anywhere — they were
# only ever printed to console or spoken by the agent, then lost once the
# call ended, even though the gatekeeper_referral/booking/objection scripts
# imply a human follow-up can act on them. Same gap that recontact_at fixed
# for callback_later.
#
# Safe to run more than once: IF NOT EXISTS makes it a no-op if a column is
# already there.
# ============================================================

import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    host='localhost', port=5432,
    database='porter_leads', user='porter',
    password=os.getenv('DB_PASSWORD', '')
)
cursor = conn.cursor()

cursor.execute("""
    ALTER TABLE lead_candidates
    ADD COLUMN IF NOT EXISTS referral_details text NULL,
    ADD COLUMN IF NOT EXISTS callback_details text NULL,
    ADD COLUMN IF NOT EXISTS objection_text text NULL
""")

conn.commit()
print("Migration applied: lead_candidates.referral_details/callback_details/objection_text (text, nullable) ensured.")

cursor.execute("""
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_name = 'lead_candidates'
    AND column_name IN ('referral_details', 'callback_details', 'objection_text')
""")
for row in cursor.fetchall():
    print(f"Verified: {row}")

cursor.close()
conn.close()
