# Porter Capital Voice Agent

A self-hosted LiveKit outbound voice agent for Porter Capital. The worker uses
the current LiveKit `AgentServer` and `AgentSession` pattern, one continuous
agent, stage-scoped function tools, and PostgreSQL through asyncpg.

## Architecture

```text
self-hosted LiveKit and SIP
        -> agent.py composition root
        -> AgentSession: Deepgram STT, OpenAI LLM, Cartesia TTS, bundled Silero VAD
           and LiveKit audio turn detection with adaptive interruption
        -> Aiva workflow tools
        -> asyncpg outcome repository
        -> structured transcript logs
```

LiveKit Cloud RTC endpoints are rejected for real jobs. Rooms, SIP, media, and
the worker stay self-hosted. Local fake jobs created by LiveKit's `console`
command skip this URL check because they do not connect to RTC. The `v1` turn
detector and adaptive interruption are the sole hosted exception and use
LiveKit Inference. Transcript turns are not stored in the database; the
existing `calls` row and final lead outcome are retained.

The workflow remains opener, pitch, qualifying, and booking. Objection and AI
disclosure temporarily interrupt and then resume the active stage. Opt-out is
terminal and writes the suppression list before the closing message.

Before each session starts, the entrypoint builds LiveKit-style context
variables for company, contact, city, state, industry, and tier and formats the
single system prompt once. Phone numbers, lead IDs, scores, internal summaries,
and website domains remain application-only data and are not added to the LLM
prompt.
Explicit requests to end or hang up use LiveKit's prebuilt `EndCallTool`, which
generates the goodbye, shuts down the session, deletes the room, and ends the
job. Opt-out requests remain on the dedicated suppression path.

LiveKit marks the prospect as away after 15 seconds of user and agent silence.
The worker uses the native `user_state_changed` event to issue three brief
check-ins at 10-second intervals. User activity cancels the check-ins; exhausted
retries close the call as `no_answer` after a short goodbye.

## Configuration

```dotenv
LIVEKIT_URL=wss://livekit.internal.example.com
LIVEKIT_API_KEY=replace_me
LIVEKIT_API_SECRET=replace_me
# Optional overrides; otherwise the LiveKit API credentials above are reused.
LIVEKIT_INFERENCE_API_KEY=replace_me
LIVEKIT_INFERENCE_API_SECRET=replace_me
AGENT_NAME=codex-agent
AGENT_VERSION=v2

DEEPGRAM_API_KEY=replace_me
OPENAI_API_KEY=replace_me
CARTESIA_API_KEY=replace_me

DATABASE_URL=postgresql://porter:password@localhost:5432/porter_leads
DB_POOL_MIN_SIZE=1
DB_POOL_MAX_SIZE=4
DB_COMMAND_TIMEOUT=10
RECONTACT_DEFAULT_DAYS=7
TEST_MODE=false

SIP_OUTBOUND_TRUNK_ID=ST_xxxxxxxxx
MY_PHONE_NUMBER=+15555550101
```

When `DATABASE_URL` is absent, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, and
`DB_PASSWORD` are used. The voice pipeline is fixed to Deepgram `nova-2`,
OpenAI `gpt-4o-mini`, the configured Cartesia voice, and LiveKit's bundled
Silero VAD. LiveKit's
full `v1` audio turn detector uses fixed endpointing from 0.3 to 2.5 seconds.
Adaptive interruptions use dedicated LiveKit Inference credentials when set,
otherwise they reuse `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`. They fall back
to VAD handling if adaptive inference is unavailable.

## Run

```bash
uv sync --locked
uv run python agent.py dev
```

Place a controlled outbound call through self-hosted LiveKit SIP:

```bash
uv run python -m scripts.call
```

For terminal-only conversation testing:

```bash
uv run agent.py console
```

Console mode uses the next eligible database lead whether or not it has a
phone number. When the database has no eligible lead, it uses a non-persistent
in-memory example lead so the conversation can still start. Real RTC jobs never
use this fixture and continue to require a database lead.

Override any console lead value directly on the command line:

```bash
uv run agent.py console \
  --company-name "Acme Manufacturing" \
  --contact-name "Jordan" \
  --city "Dallas" \
  --state "Texas" \
  --industry "manufacturing" \
  --tier "Warm" \
  --phone "+15555550101" \
  --website-domain "acme.example" \
  --current-score 88.5 \
  --why-now-summary "Testing the console workflow" \
  --naics-code "332710"
```

All lead options are optional. Omit `--phone` to test without a number.
`--lead-candidate-id` is also accepted for context testing, but console fixture
calls are never persisted. LiveKit console options such as `--text`,
`--input-device`, and `--output-device` can be used in the same command. Passing
any lead option makes the configured fixture override a database lead for that
console run.

## Database and logs

Apply the complete schema from the ordered SQL files in `migrations/`:

```bash
uv run python -m scripts.run_migrations
```

The asyncpg runner applies each pending migration in its own transaction and
records its filename and SHA-256 checksum in `schema_migrations`. Applied files
are skipped, and changing an applied file causes the runner to stop. Add a new
numbered SQL file for every future schema change. The migrations contain no
seed data and never drop historical transcript tables.

All runtime and administrative connections use asyncpg. Call completion and
lead outcome updates share one transaction. Suppression entries also share one
transaction.

Each finalized user or assistant turn is logged at INFO with `room`, `role`,
`stage`, and `text`. These logs contain personal and commercial information;
configure access, encryption, redaction, and retention in the log platform.

## Validate

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
uv run python -m py_compile agent.py app/config.py app/agent/*.py app/infrastructure/*.py
```

Implementation choices follow the official
[LiveKit Python examples](https://github.com/livekit-examples/python-agents-examples),
[LiveKit context variables recipe](https://docs.livekit.io/reference/recipes/context_variables/),
[LiveKit prompting guide](https://docs.livekit.io/agents/start/prompting/), and
[asyncpg usage guide](https://magicstack.github.io/asyncpg/current/usage.html).

See [the refactor notes](docs/REFACTOR.md) for rationale and production
recommendations.
