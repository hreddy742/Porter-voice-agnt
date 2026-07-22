CREATE TABLE IF NOT EXISTS lead_candidates (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id        uuid NOT NULL REFERENCES companies(id),
    status            text NOT NULL DEFAULT 'active',
    sector_excluded   boolean NOT NULL DEFAULT false,
    sales_status      text NOT NULL DEFAULT 'research',
    tier              text,
    current_score     double precision,
    why_now_summary   text,
    contact_name      text,
    recontact_at      timestamptz,
    referral_details text,
    callback_details text,
    objection_text    text,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now(),
    deleted_at        timestamptz
);

ALTER TABLE lead_candidates
    ADD COLUMN IF NOT EXISTS contact_name text,
    ADD COLUMN IF NOT EXISTS recontact_at timestamptz,
    ADD COLUMN IF NOT EXISTS referral_details text,
    ADD COLUMN IF NOT EXISTS callback_details text,
    ADD COLUMN IF NOT EXISTS objection_text text;

CREATE INDEX IF NOT EXISTS idx_lead_candidates_company_id
    ON lead_candidates (company_id);

CREATE INDEX IF NOT EXISTS idx_lead_candidates_call_queue
    ON lead_candidates (status, sales_status, sector_excluded, tier, current_score DESC)
    WHERE status = 'active' AND sector_excluded = false;
