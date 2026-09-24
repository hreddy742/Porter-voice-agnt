# Porter Capital Voice Agent (Aiva)

Outbound cold-calling voice agent for Porter Capital invoice factoring.
Aiva places real phone calls, runs the conversation, qualifies the
prospect, and books a callback with a human advisor — or logs why it
didn't — writing the outcome back to a Postgres lead database.

Built with [LiveKit Agents](https://docs.livekit.io/agents/) (voice
pipeline + tool-calling), Twilio (outbound telephony), Deepgram (STT),
Cartesia/ElevenLabs (TTS), and Postgres (lead intelligence).

## Why this exists

Cold-calling a lead list at any real volume needs a human on the phone
until the point a prospect is actually qualified and ready to book — this
project explores how much of that first-pass qualification can be handled
by a voice agent that reasons about the conversation in real time, rather
than a static IVR script.

## Architecture

```text
call.py                         # 1. select a lead, dial their real number
  └─ app/calling/dialer.py      #    creates a LiveKit room tagged with
                                 #    that lead's id, dials via Twilio

agent.py (worker, always running)
  └─ app/agent/worker.py        # 2. joins the room, reads the tagged
                                 #    lead id, loads that SAME lead
      └─ app/agent/supervisor.py   # 3. PorterSupervisor (LiveKit Agent)
          └─ app/agent/tasks/*      #    routes the call through tasks —
                                     #    opener → pitch → objection →
                                     #    qualifier → booking/exit — via
                                     #    LLM tool calls, not a fixed script
      └─ app/db/                # 4. call outcome written back to Postgres
```

Steps 1 and 2 used to be decoupled — the dialer rang a fixed test number
while the worker independently picked whichever lead was "next" in the
database, so the number Twilio actually called and the lead the agent
believed it was talking to could silently mismatch. The dialer now
selects the lead first, dials *that* lead's real number, and tags the
LiveKit room with its id so the worker uses the exact same lead.

## Agent design

`PorterSupervisor` (`app/agent/supervisor.py`) is a LiveKit `Agent` whose
system prompt describes a call-flow routing table, and whose methods are
exposed as `function_tool()`s: `start_call`, `run_pitch`, `run_objection`,
`run_qualifier`, `run_booking`, `run_disclosure`, `run_exit`. The LLM
chooses which tool to call based on the live conversation — an objection
can be raised and routed from any point in the call, not just a fixed
slot in a script — and call state (`call_result`, the active lead)
persists across the whole session until it's written to Postgres.

Guardrails are enforced in the prompt (e.g. never quoting rates) and
backstopped in code — `app/agent/speech.py` strips any raw tool-call
syntax or internal routing text that leaks into what the model would
otherwise say out loud, independent of whether the model followed the
prompt correctly.

## LLM & TTS routing

`app/agent/providers.py` abstracts model selection behind `LLM_PROVIDER`:
Grok (via xAI direct or LiveKit Inference), Claude, GPT-4o-mini, Groq, or
a local Ollama model. An optional `LLM_FAILOVER_PROVIDER` wraps the
primary in a `FallbackAdapter`, so a mid-call provider outage doesn't
kill the conversation — `app/agent/worker.py`'s `prewarm_qwen` pre-loads
the local fallback model at worker startup specifically to avoid a ~24s
cold-load turning into dead air on a live call the first time failover
is needed.

## Compliance

- Aiva discloses she's an AI when asked, and always by the second turn of
  the call (`app/agent/tasks/opener.py`, `run_disclosure`).
- Opt-outs ("remove me from your list") are logged to an append-only
  suppression table, checked by name *and* domain, before any future call
  (`app/db/leads.py`).

## Known limitations

- **No evaluation harness.** Call outcomes are written to Postgres but
  nothing yet aggregates booking rate, objection-handling success, or
  false-hangup rate — there's no data-backed claim of *how well* this
  performs, only that it runs.
- **No production observability.** `opentelemetry` is a dependency but
  isn't wired up; logging is `print()`. Debugging a bad call after the
  fact currently means reading console output, not traces.
- **No campaign/concurrency layer.** `call.py` places one call at a time,
  manually. There's no queueing, pacing, retry/backoff, or concurrency
  limiting for dialing a lead list at volume.
- **No adversarial testing of the tool-routing prompt.** The LLM decides
  which tool to call based on live, untrusted speech from whoever answers
  the phone; `speech.py` cleans up leaked tool-call text after generation
  but nothing prevents a caller from talking the model outside its
  intended routing table.

## Layout

```text
app/
  agent/          # LiveKit worker, supervisor, tasks, LLM/TTS providers
  calling/        # Twilio + LiveKit outbound dialer
  db/             # Lead selection and status updates
  config.py       # pydantic Settings from .env
  knowledge/      # Porter Capital product copy
scripts/          # Ops / smoke-test helpers
agent.py          # Thin wrapper → python agent.py console|dev|start
call.py           # Thin wrapper → python call.py
```

## Setup

```bash
cp .env.example .env   # fill in DB / LiveKit / Twilio / LLM / TTS credentials
uv sync                # or: pip install -e .
python scripts/init_porter_schema.py   # first time only — creates tables + a test lead
```

## Run

```bash
# Terminal 1 — start the agent worker
python agent.py dev

# Terminal 2 — place an outbound call
python call.py
```

With `TEST_MODE=true` in `.env`, `call.py` still selects a real lead from
the database (so the agent gets real context) but dials `MY_PHONE_NUMBER`
instead of the lead's actual number, and skips writing the call result
back to Postgres.

Local playground (no phone call at all):

```bash
python agent.py console
```

## Scripts

```bash
python scripts/init_porter_schema.py   # first time only — create tables + seed test lead
python scripts/test_db.py
python scripts/check_db.py
python scripts/reset_lead.py
```
