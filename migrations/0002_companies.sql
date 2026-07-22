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

CREATE INDEX IF NOT EXISTS idx_companies_website_domain
    ON companies (website_domain)
    WHERE website_domain IS NOT NULL;
