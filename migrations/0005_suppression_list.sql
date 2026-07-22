CREATE TABLE IF NOT EXISTS suppression_list (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    match_type   text NOT NULL,
    match_value  text NOT NULL,
    reason       text,
    source       text,
    active       boolean NOT NULL DEFAULT true,
    created_at   timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT suppression_list_match_type_match_value_key
        UNIQUE (match_type, match_value)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_suppression_list_match
    ON suppression_list (match_type, match_value);

CREATE INDEX IF NOT EXISTS idx_suppression_list_active_match
    ON suppression_list (active, match_type, match_value);
