# ============================================================
# db.py — Database connection for Porter Capital Voice Agent
# ============================================================
# This file does seven things:
#   1. get_connection()      → open a DB connection
#   2. create_call()         → start a call record, return its ID
#   3. add_call_turn()       → persist one transcript turn
#   4. finish_call()         → mark a call record complete
#   5. get_next_lead()       → find the best lead to call
#   6. update_lead_status()  → write call result back to DB
#   7. add_to_suppression()  → log opt-outs immediately
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


def create_call(lead_candidate_id, agent_version, llm_provider,
                tts_provider, room_id):
    """Create a call record and return its database ID."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO calls (
                    lead_candidate_id,
                    call_started_at,
                    agent_version,
                    llm_provider,
                    tts_provider,
                    room_id
                )
                VALUES (%s, now(), %s, %s, %s, %s)
                RETURNING id
            """, (
                lead_candidate_id,
                agent_version,
                llm_provider,
                tts_provider,
                room_id,
            ))
            return cursor.fetchone()[0]


def add_call_turn(call_id, turn_number, role, text, task_or_stage):
    """Persist one ordered user or assistant transcript turn."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO call_turns (
                    call_id,
                    turn_number,
                    role,
                    text,
                    task_or_stage
                )
                VALUES (%s, %s, %s, %s, %s)
            """, (call_id, turn_number, role, text, task_or_stage))


def finish_call(call_id, final_call_result):
    """Mark a call complete with the same result used for the lead."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE calls
                SET call_ended_at = now(), final_call_result = %s
                WHERE id = %s
            """, (final_call_result, call_id))


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
            lc.why_now_summary,
            lc.contact_name
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
        "contact_name":      row[11],
    }


def update_lead_status(lead_candidate_id, new_status, referral_details=None,
                        callback_details=None, objection_text=None):
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

    referral_details / callback_details / objection_text: optional free-text
    columns (see migrate_add_call_notes_columns.py). Only written when
    provided (non-None) — omitting one never nulls out a value written by a
    previous call.
    """

    conn   = get_connection()
    cursor = conn.cursor()

    set_clauses = ["sales_status = %s", "updated_at = now()"]
    params      = [new_status]

    if new_status == 'callback_later':
        # recontact_at added via migrate_add_recontact_at.py. The interval is
        # a placeholder (RECONTACT_DEFAULT_DAYS, default 7) pending a real
        # business decision on cadence — see db.py's RECONTACT_DEFAULT_DAYS.
        set_clauses.append("recontact_at = now() + (%s || ' days')::interval")
        params.append(RECONTACT_DEFAULT_DAYS)

    if referral_details is not None:
        set_clauses.append("referral_details = %s")
        params.append(referral_details)
    if callback_details is not None:
        set_clauses.append("callback_details = %s")
        params.append(callback_details)
    if objection_text is not None:
        set_clauses.append("objection_text = %s")
        params.append(objection_text)

    params.append(lead_candidate_id)

    cursor.execute(f"""
        UPDATE lead_candidates
        SET {', '.join(set_clauses)}
        WHERE id = %s
    """, params)

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
