CREATE TABLE IF NOT EXISTS company_contactability (
    id                     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_candidate_id      uuid NOT NULL REFERENCES lead_candidates(id),
    company_id             uuid REFERENCES companies(id),
    phone                  text,
    contactability_status  text NOT NULL DEFAULT 'valid',
    created_at             timestamptz NOT NULL DEFAULT now(),
    updated_at             timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_company_contactability_lead_candidate_id
    ON company_contactability (lead_candidate_id);

CREATE INDEX IF NOT EXISTS idx_company_contactability_company_id
    ON company_contactability (company_id);

CREATE INDEX IF NOT EXISTS idx_company_contactability_callable
    ON company_contactability (contactability_status, lead_candidate_id)
    WHERE phone IS NOT NULL;
