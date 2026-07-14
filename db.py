# ============================================================
# db.py — Database connection for Porter Capital Voice Agent
# ============================================================
# This file does three things:
#   1. get_next_lead()       → find the best lead to call
#   2. update_lead_status()  → write call result back to DB
#   3. add_to_suppression()  → log opt-outs immediately
#
# Connected to: porter_leads database (your Lead Intelligence DB)
# ============================================================

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# --- Database connection settings ---
# These match your porter_leads database exactly
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "porter_leads",
    "user":     "porter",
    "password": os.getenv("DB_PASSWORD", ""),
}

# How far out to schedule a recontact when a call ends in 'callback_later'
# (bad timing, not a genuine no). 7 days is a placeholder — the right
# interval is a business decision John hasn't confirmed yet. Override via
# env var rather than editing this default in code once that's settled.
RECONTACT_DEFAULT_DAYS = int(os.getenv("RECONTACT_DEFAULT_DAYS", "7"))


def get_connection():
    """
    Opens a connection to the porter_leads database.
    Think of this like picking up the phone to call the database.
    Every function below calls this first, then closes it when done.
    """
    return psycopg2.connect(**DB_CONFIG)


def get_next_lead():
    """
    Finds the single best lead to call right now.

    Rules applied in order:
    1. Must be active (not deleted or archived)
    2. Must not be a sector exclusion (no trucking/construction)
    3. Must have a phone number
    4. Phone must be valid (not disconnected or invalid)
    5. Must not be on the suppression list (do not call list)
    6. Ordered by tier: Hot first, then Warm, then Cold
    7. Within same tier, higher score goes first

    Returns a dictionary with all lead details,
    or None if no leads are available.

    NOTE: does not yet consider recontact_at. sales_status filtering
    ('research' only) already excludes 'callback_later' leads, so this is
    not a live bug — but once John confirms the real recontact cadence,
    this should also allow sales_status = 'callback_later' leads back in
    once recontact_at <= now().
    """

    query = """
        SELECT
            c.canonical_name        AS company_name,
            c.city,
            c.state,
            c.industry,
            c.naics_code,
            c.website_domain,
            cc.phone,
            lc.id                   AS lead_candidate_id,
            lc.tier,
            lc.current_score,
            lc.why_now_summary
        FROM lead_candidates lc

        -- Join to get company details (name, location, industry)
        JOIN companies c
            ON c.id = lc.company_id

        -- Join to get contact details (phone number)
        JOIN company_contactability cc
            ON cc.lead_candidate_id = lc.id

        WHERE
            -- Only active leads
            lc.status = 'active'

            -- Skip excluded sectors (trucking, construction, etc.)
            AND lc.sector_excluded = false

            -- Only leads we have not contacted yet
            AND lc.sales_status = 'research'

            -- Must have a phone number
            AND cc.phone IS NOT NULL

            -- Phone must be reachable
            AND cc.contactability_status NOT IN ('invalid', 'disconnected')

            -- Must NOT be on the suppression list
            AND NOT EXISTS (
                SELECT 1 FROM suppression_list sl
                WHERE sl.active = true
                AND (
                    (sl.match_type = 'company_name'
                     AND sl.match_value = c.canonical_name)
                    OR
                    (sl.match_type = 'domain'
                     AND sl.match_value = c.website_domain)
                )
            )

        -- Best leads first: Hot > Warm > Cold, then highest score
        ORDER BY
            CASE lc.tier
                WHEN 'Hot'  THEN 1
                WHEN 'Warm' THEN 2
                WHEN 'Cold' THEN 3
                ELSE 4
            END,
            lc.current_score DESC

        -- Only get the single best lead
        LIMIT 1
    """

    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    # No leads available right now
    if row is None:
        return None

    # Return as a clean dictionary so the agent can read it easily
    return {
        "company_name":      row[0],
        "city":              row[1],
        "state":             row[2],
        "industry":          row[3],
        "naics_code":        row[4],
        "website_domain":    row[5],
        "phone":             row[6],
        "lead_candidate_id": str(row[7]),
        "tier":              row[8],
        "current_score":     row[9],
        "why_now_summary":   row[10],
    }


def update_lead_status(lead_candidate_id, new_status):
    """
    After a call ends, write the outcome back to the database.
    This keeps your Lead Intelligence system up to date.

    Possible statuses:
    - 'contacted'       → we called, they answered, general contact made
    - 'interested'      → they want to know more about Porter Capital
    - 'not_interested'  → they said no
    - 'callback_booked' → they agreed to speak with a human rep
    - 'callback_later'  → soft no / bad timing — try again down the road.
                           Also sets recontact_at = now() + RECONTACT_DEFAULT_DAYS.
    - 'suppressed'      → they asked to be removed from our list
    - 'no_answer'       → nobody picked up
    - 'sector_excluded' → turned out to be wrong industry mid-call
    """

    conn   = get_connection()
    cursor = conn.cursor()

    if new_status == 'callback_later':
        # recontact_at added via migrate_add_recontact_at.py. The interval is
        # a placeholder (RECONTACT_DEFAULT_DAYS, default 7) pending a real
        # business decision on cadence — see db.py's RECONTACT_DEFAULT_DAYS.
        cursor.execute("""
            UPDATE lead_candidates
            SET
                sales_status = %s,
                recontact_at = now() + (%s || ' days')::interval,
                updated_at   = now()
            WHERE id = %s
        """, (new_status, RECONTACT_DEFAULT_DAYS, lead_candidate_id))
    else:
        cursor.execute("""
            UPDATE lead_candidates
            SET
                sales_status = %s,
                updated_at   = now()
            WHERE id = %s
        """, (new_status, lead_candidate_id))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"Lead {lead_candidate_id} updated to: {new_status}")


def add_to_suppression(company_name, website_domain, reason="opted_out"):
    """
    When a prospect says 'remove me from your list' or 'stop calling',
    add them to the suppression list immediately.

    This is a compliance requirement:
    - Append only — entries are never deleted
    - Checked before every future call
    - Logged with reason and source

    We add two entries: one by company name, one by domain.
    This catches the company even if their name is spelled differently next time.
    """

    conn   = get_connection()
    cursor = conn.cursor()

    # Add by company name
    cursor.execute("""
        INSERT INTO suppression_list
            (match_type, match_value, reason, source)
        VALUES
            ('company_name', %s, %s, 'voice_agent')
        ON CONFLICT DO NOTHING
    """, (company_name, reason))

    # Add by domain if we have it
    if website_domain:
        cursor.execute("""
            INSERT INTO suppression_list
                (match_type, match_value, reason, source)
            VALUES
                ('domain', %s, %s, 'voice_agent')
            ON CONFLICT DO NOTHING
        """, (website_domain, reason))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"Added to suppression list: {company_name} | Reason: {reason}")
