# ============================================================
# init_porter_schema.py — create lead-intelligence tables + seed
# ============================================================
# The local `porter` DB exists but has no tables. This script creates
# the relations that app/db/leads.py and the ops scripts expect, then
# inserts one callable test lead (Apex Staffing Solutions).
# Safe to re-run: CREATE IF NOT EXISTS + ON CONFLICT seed.
# ============================================================

from __future__ import annotations

from app.db import get_connection

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS companies (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    canonical_name  text NOT NULL UNIQUE,
    city            text,
    state           text,
    industry        text,
    naics_code      text,
    website_domain  text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lead_candidates (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id        uuid NOT NULL REFERENCES companies(id),
    status            text NOT NULL DEFAULT 'active',
    sector_excluded   boolean NOT NULL DEFAULT false,
    sales_status      text NOT NULL DEFAULT 'research',
    tier              text,
    current_score     double precision,
    why_now_summary   text,
    recontact_at      timestamptz NULL,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now(),
    deleted_at        timestamptz NULL
);

CREATE TABLE IF NOT EXISTS company_contactability (
    id                      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_candidate_id       uuid NOT NULL REFERENCES lead_candidates(id),
    company_id              uuid REFERENCES companies(id),
    phone                   text,
    contactability_status   text NOT NULL DEFAULT 'valid',
    created_at              timestamptz NOT NULL DEFAULT now(),
    updated_at              timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS suppression_list (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    match_type   text NOT NULL,
    match_value  text NOT NULL,
    reason       text,
    source       text,
    active       boolean NOT NULL DEFAULT true,
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (match_type, match_value)
);
"""

SEED_SQL = """
WITH company AS (
    INSERT INTO companies (canonical_name, city, state, industry, naics_code, website_domain)
    VALUES (
        'Apex Staffing Solutions',
        'Austin',
        'TX',
        'staffing',
        '561320',
        'apexstaffing.example'
    )
    ON CONFLICT (canonical_name) DO UPDATE
        SET updated_at = now()
    RETURNING id
),
existing_lead AS (
    SELECT lc.id, lc.company_id
    FROM lead_candidates lc
    JOIN companies c ON c.id = lc.company_id
    WHERE c.canonical_name = 'Apex Staffing Solutions'
    LIMIT 1
),
lead AS (
    INSERT INTO lead_candidates (
        id, company_id, status, sector_excluded, sales_status,
        tier, current_score, why_now_summary
    )
    SELECT
        '62c8587c-0286-4e41-a6e9-43557125ed87'::uuid,
        company.id,
        'active',
        false,
        'research',
        'Hot',
        92.5,
        'Growing staffing firm with B2B invoices — good factoring fit.'
    FROM company
    WHERE NOT EXISTS (SELECT 1 FROM existing_lead)
    RETURNING id, company_id
),
resolved AS (
    SELECT id, company_id FROM lead
    UNION ALL
    SELECT id, company_id FROM existing_lead
)
INSERT INTO company_contactability (lead_candidate_id, company_id, phone, contactability_status)
SELECT
    resolved.id,
    resolved.company_id,
    '+16592532045',
    'valid'
FROM resolved
WHERE NOT EXISTS (
    SELECT 1 FROM company_contactability cc
    WHERE cc.lead_candidate_id = resolved.id
);
"""


def main() -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(SCHEMA_SQL)
    cursor.execute(SEED_SQL)
    conn.commit()

    cursor.execute("""
        SELECT c.canonical_name, lc.sales_status, lc.tier, cc.phone
        FROM lead_candidates lc
        JOIN companies c ON c.id = lc.company_id
        JOIN company_contactability cc ON cc.lead_candidate_id = lc.id
        WHERE c.canonical_name = 'Apex Staffing Solutions'
    """)
    row = cursor.fetchone()
    print("Schema ready.")
    if row:
        print(f"Seed lead: {row[0]} | status={row[1]} | tier={row[2]} | phone={row[3]}")
    else:
        print("WARNING: seed lead not found after insert.")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
