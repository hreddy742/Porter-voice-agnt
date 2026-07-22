CREATE TABLE IF NOT EXISTS calls (
    id                 serial PRIMARY KEY,
    lead_candidate_id  uuid NOT NULL REFERENCES lead_candidates(id),
    call_started_at    timestamptz NOT NULL DEFAULT now(),
    call_ended_at      timestamptz,
    agent_version      text,
    llm_provider       text,
    tts_provider       text,
    final_call_result  text,
    room_id            text
);

CREATE INDEX IF NOT EXISTS idx_calls_lead_candidate_id
    ON calls (lead_candidate_id);

CREATE INDEX IF NOT EXISTS idx_calls_room_id
    ON calls (room_id)
    WHERE room_id IS NOT NULL;
