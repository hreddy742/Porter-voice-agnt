# Refactor notes

## What changed

- `agent.py` is the composition root and uses LiveKit `AgentServer`, an RTC
  session decorator, prewarmed bundled Silero VAD, `AgentSession.start`, and
  `cli.run_app(server)`.
- Deepgram STT, OpenAI LLM, and Cartesia TTS are constructed directly in the
  entrypoint. There is no provider factory, fallback, or runtime selector.
- `Aiva` now contains only workflow state, tool exposure, result validation,
  deterministic persistence decisions, and call termination. Custom LLM, TTS,
  and transcription nodes and all output regex filtering were removed.
- One system prompt owns identity, speech style, interruption behavior, tool
  rules, workflow classifications, approved product knowledge, and closings.
- The RTC entrypoint derives six prompt-safe context variables from each lead
  and passes them into the agent. The system prompt is formatted once before
  session startup; operational lead fields remain outside the LLM context.
- Greetings and closings use `AgentSession.generate_reply`. Closings wait for
  playout before LiveKit room deletion.
- PostgreSQL uses asyncpg exclusively. Runtime calls use a small pool, native
  positional parameters, and transactions for grouped writes.
- PostgreSQL schema management uses immutable, ordered SQL files in
  `migrations/`. One asyncpg runner tracks SHA-256 checksums and applies each
  pending file in its own transaction without seeding or deleting data.
- Transcript persistence and retrieval were removed. Turns are emitted to the
  application logger and existing historical transcript tables are not dropped.
- Inactivity uses LiveKit's `user_away_timeout` and `user_state_changed` event.
  A session-local cancellable task spaces reminders and reuses the normal
  `no_answer` closing and database-finalization path.
- Turn handling uses LiveKit's full `v1` audio turn detector, fixed 0.3-to-2.5
  second endpointing, adaptive interruption, false-interruption recovery, and
  preemptive generation without preemptive TTS. Deepgram supplies the required
  word-aligned transcripts and bundled Silero VAD remains the interruption fallback.
- Explicit hangup requests use LiveKit's beta `EndCallTool`. Its callback sets
  the business result before the standard shutdown callback persists the call;
  the tool owns goodbye playout, session shutdown, room deletion, and job exit.

## Why

The previous controller duplicated prompt policy in regex filters and custom
model nodes. That increased latency and coupled business behavior to streamed
model internals. The replacement follows LiveKit's normal extension points:
prompt instructions, function tools, session events, and lifecycle callbacks.

The database repository is asynchronous end to end, so database access no
longer blocks the voice event loop or requires thread wrappers.

## Operational recommendations

- Pin and test LiveKit plugin versions together before upgrading.
- Keep the worker close to the self-hosted LiveKit, STT, LLM, and TTS services.
- Alert on session errors, database pool exhaustion, call duration, tool errors,
  and missing final outcomes.
- Treat transcript logs as sensitive data and enforce short retention and
  role-based access.
- Monitor LiveKit Inference availability and credential failures separately
  from the self-hosted RTC service. Only turn detection and adaptive
  interruption use this hosted dependency.
- Exercise the complete call matrix with recorded test audio before production
  rollout, especially opt-out, interruption, unclear digits, and callback time
  confirmation.
