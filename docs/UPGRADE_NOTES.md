# Upgrade notes

The worker now targets `livekit-agents` 1.6.4 and its `AgentServer` API. The
launch command is `uv run python agent.py dev`.

Required migration steps:

1. Install the updated requirements, including asyncpg.
2. Remove `BOUNDARY_AUDIO_MANIFEST`; prerecorded boundary assets are no longer
   used.
3. Configure `DATABASE_URL`, or the documented individual database variables.
4. Remove `LLM_PROVIDER`, `TTS_PROVIDER`, `LLM_FAILOVER_PROVIDER`, and
   credentials used only by removed provider integrations.
5. Configure `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, and `CARTESIA_API_KEY`.
6. Route INFO logs to the approved transcript log destination and apply its
   retention policy.
7. Apply the ordered SQL migrations with
   `uv run python -m scripts.run_migrations` before starting the worker.
8. Do not drop an existing `call_turns` table during rollout. It is left intact
   for historical compatibility but receives no new writes.

No lead status, suppression, referral, objection, callback, or call-outcome
columns were renamed.
