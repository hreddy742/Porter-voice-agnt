CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    text PRIMARY KEY,
    checksum    text NOT NULL CHECK (length(checksum) = 64),
    applied_at  timestamptz NOT NULL DEFAULT now()
);
