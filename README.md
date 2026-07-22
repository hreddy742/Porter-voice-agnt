# Porter Capital Voice Agent

Outbound AI voice agent for Porter Capital, built with LiveKit Agents. The
agent selects a lead from PostgreSQL, handles a staged sales conversation,
records the transcript, saves the outcome, and supports human callback booking.

## Which worker should I use?

Use `agent_v2.py`. It is the canonical worker and registers in LiveKit as
`codex-agent`.

`agent.py` is the older implementation. It remains in the repository for
comparison and regression coverage, but it should not be used to start the
current worker.

## How a call works

```text
Phone caller
    -> LiveKit room
    -> Silero voice activity detection
    -> Deepgram speech-to-text
    -> Aiva conversation-stage controller
    -> configured LLM (GPT by default, optional fallback)
    -> Cartesia text-to-speech
    -> Phone caller

During the call
    -> PostgreSQL stores ordered transcript turns
    -> PostgreSQL stores the final lead status and follow-up details
```

The conversation progresses through `hello`, `opener`, `pitch`, `qualifying`,
optional `objection` or `disclosure`, and `booking`. Function tools own their
responses explicitly so LiveKit does not generate duplicate follow-up replies.

## Repository layout

```text
.
|-- agent_v2.py                 Canonical LiveKit worker
|-- agent.py                    Legacy worker retained for comparison
|-- db.py                       Lead selection and outcome persistence
|-- knowledge.py                Approved Porter Capital product knowledge
|-- call_transcript.py          Asynchronous transcript persistence
|-- requirements.txt            Pinned Python dependencies
|-- docs/
|   `-- UPGRADE_NOTES.md        Engineering decisions and known risks
|-- scripts/
|   |-- call.py                 Controlled Twilio test-call launcher
|   |-- simulate.py             Fourteen-scenario behavior simulation
|   |-- run_big_batch.py        One-hundred-scenario evaluation
|   |-- migrate_*.py            Idempotent database migrations
|   |-- check_*.py              Database and provider diagnostics
|   `-- reset_*.py              Database maintenance commands
`-- tests/
    |-- test_*.py               Deterministic regression checks
    `-- manual/                 Checks that contact live providers
```

Generated logs, simulation output, Python caches, local tool settings, and
`.env` are ignored by Git.

## Prerequisites

- Windows, macOS, or Linux with Python 3.11 or newer
- A LiveKit project
- PostgreSQL running locally on port `5432`
- An existing `porter_leads` database owned by or accessible to user `porter`
- Provider accounts for Deepgram, the selected LLM, and the selected TTS
- Twilio only when using the phone test-call launcher

The repository contains additive migrations for the voice-agent fields and
transcript tables. It does not contain the original lead-intelligence database
schema (`companies`, `lead_candidates`, `company_contactability`, and
`suppression_list`). That base schema must already exist.

## 1. Create the Python environment

From the project root:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source venv/bin/activate`.

## 2. Configure `.env`

Create `.env` in the project root. Never commit this file or paste real keys
into documentation, issues, or logs.

```dotenv
# LiveKit worker
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=replace_me
LIVEKIT_API_SECRET=replace_me

# Speech recognition
DEEPGRAM_API_KEY=replace_me

# Language model
LLM_PROVIDER=gpt
OPENAI_API_KEY=replace_me
LLM_FAILOVER_PROVIDER=groq
GROQ_API_KEY=replace_me

# Speech generation
TTS_PROVIDER=cartesia
CARTESIA_API_KEY=replace_me

# Local database: host, port, database, and user are currently fixed in db.py
DB_PASSWORD=replace_me

# Keep this true while validating; production writes require false
TEST_MODE=true
AGENT_VERSION=v2

# Required only by: python -m scripts.call
TWILIO_ACCOUNT_SID=replace_me
TWILIO_AUTH_TOKEN=replace_me
TWILIO_PHONE_NUMBER=+15555550100
MY_PHONE_NUMBER=+15555550101
LIVEKIT_TWILIO_WS_URL=wss://your-project.sip.livekit.cloud/twilio
```

Provider choices supported by `agent_v2.py`:

- `LLM_PROVIDER`: `gpt`, `groq`, `grok`, `qwen`, `mistral-local`, or `gptoss`
- `TTS_PROVIDER`: `cartesia`, `elevenlabs`, `kokoro`, or `chatterbox`

Local providers require their corresponding local service URL and model. See
[`docs/UPGRADE_NOTES.md`](docs/UPGRADE_NOTES.md) before changing providers or
latency settings.

## 3. Prepare the database

The application currently connects to:

```text
host: localhost
port: 5432
database: porter_leads
user: porter
password: DB_PASSWORD from .env
```

After the base lead database exists, run the idempotent migrations from the
project root:

```powershell
python -m scripts.migrate_add_contact_name
python -m scripts.migrate_add_recontact_at
python -m scripts.migrate_add_calls_tables
python -m scripts.migrate_add_call_notes_columns
```

Each migration uses `IF NOT EXISTS`, prints what it changed, and verifies the
result. Check that a callable lead is available without changing data:

```powershell
python -m scripts.check_next_lead
```

## 4. Start the worker

For local development:

```powershell
python agent_v2.py dev
```

For a production process managed by your deployment platform:

```powershell
python agent_v2.py start
```

The worker is ready when the log includes:

```text
registered worker {"agent_name": "codex-agent", ...}
```

Keep this terminal running. In the LiveKit console, look for the agent name
`codex-agent`.

## 5. Make a controlled test call

With the worker running in one terminal, open another terminal in the project
root and run:

```powershell
python -m scripts.call
```

The script creates a LiveKit room and asks Twilio to call `MY_PHONE_NUMBER`.
It is a controlled test launcher, not a production campaign dialer.

Recommended validation sequence:

1. Set `TEST_MODE=true`.
2. Start `agent_v2.py dev`.
3. Confirm `codex-agent` registers in LiveKit.
4. Run `python -m scripts.call`.
5. Review the transcript, stage transitions, and `[TURN LATENCY]` log lines.
6. Confirm the call row and ordered turns were stored.
7. Set `TEST_MODE=false` only after the behavior is approved.

With `TEST_MODE=true`, suppression-list and final lead-status writes are
skipped. Call and transcript records are still created, so use a test database
or test lead when complete isolation is required.

## Automated checks

Run these from the project root after changing agent behavior:

```powershell
python -m unittest tests.test_function_tool_response_ownership
python -m tests.test_project_layout
python -m tests.test_deterministic_transition_speech
python -m tests.test_opener_result_enforcement
python -m tests.test_callback_intent_routing
python -m tests.test_drift_correction
python -m tests.test_turn_latency_logging
python -m tests.test_rate_guardrail
```

The first check protects the single-response-owner rule for every LiveKit
function tool. The remaining checks cover stage progression, callback routing,
opener enforcement, drift correction, latency logging, and rate claims.

## Manual and simulation checks

Files under `tests/manual/` can contact configured LLM providers. Run them
individually only when live-provider usage is intended, for example:

```powershell
python -m tests.manual.test_opener_cleanpath_live_manual
python -m tests.manual.test_callback_intent_routing_live_manual
```

Run the standard simulation set:

```powershell
python -m scripts.simulate
```

Run one named scenario:

```powershell
python -m scripts.simulate --scenario compound_1
```

Run the larger evaluation:

```powershell
python -m scripts.run_big_batch
```

Simulation results are written under `simulation_runs/` and are intentionally
not committed.

## Operator scripts

| Command | Purpose | Data impact |
|---|---|---|
| `python -m scripts.check_next_lead` | Show the next eligible lead | Read-only |
| `python -m scripts.check_db` | Inspect database state | Read-only |
| `python -m scripts.check_test_lead` | List callable leads | Read-only |
| `python -m scripts.check_voice` | Verify Cartesia audio generation | Calls provider |
| `python -m scripts.reset_lead` | Reset the script's selected lead | Writes database |
| `python -m scripts.reset_any_lead` | Reset matching contacted leads | Writes database |

Review both reset scripts before running them. They are maintenance tools, not
normal worker startup steps.

## Troubleshooting

### The agent is not visible in LiveKit

- Confirm the command is `python agent_v2.py dev`.
- Confirm `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` belong to
  the same LiveKit project.
- Look for `registered worker` and `agent_name: codex-agent` in the terminal.
- Make sure another worker with the same name is not consuming the job.

### The worker says `No leads available`

- Run `python -m scripts.check_next_lead`.
- Confirm the lead is active, has `sales_status='research'`, has a usable phone
  number, is not sector-excluded, and is not suppressed.

### The worker starts but produces no speech

- Verify the Deepgram and selected TTS keys.
- Run `python -m scripts.check_voice` when using Cartesia.
- Confirm the selected `LLM_PROVIDER` has its required API key or local service.
- Check the worker's `error` event and provider timeout logs.

### Calls feel slow

- Inspect `[TURN LATENCY]` lines before changing endpointing or provider
  settings.
- Separate endpoint/VAD delay, transcription delay, LLM delay, and TTS delay.
- Do not tune multiple components at once; validate one measured cause at a
  time.

## Production checklist

- Use `agent_v2.py`, not `agent.py`.
- Run all deterministic checks.
- Perform a controlled live-provider call.
- Confirm database backups and migration results.
- Confirm opt-out and suppression behavior.
- Set `TEST_MODE=false` only for approved production writes.
- Store secrets in the deployment platform, never in Git.
- Run one managed worker instance per intended capacity configuration.
- Monitor provider failures, turn latency, call outcomes, and transcript writes.

Engineering history, validated fixes, and known upstream LiveKit risks are in
[`docs/UPGRADE_NOTES.md`](docs/UPGRADE_NOTES.md).
