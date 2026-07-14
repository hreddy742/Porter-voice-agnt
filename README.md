# Porter Capital Voice Agent (Aiva)

Outbound cold-calling voice agent for Porter Capital invoice factoring.
Built with LiveKit Agents, Twilio, Deepgram, and Postgres (`porter` by default).

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

## Run

```bash
# Terminal 1 — start the agent worker
python agent.py dev

# Terminal 2 — place an outbound call
python call.py
```

Local playground (no phone):

```bash
python agent.py console
```

Configure secrets in `.env` at the repo root. Database connection is loaded via
pydantic Settings (`app/config.py`):

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=porter
DB_USER=postgres
DB_PASSWORD=postgres
```

## Scripts

```bash
# First time only — create tables + seed Apex Staffing test lead
python scripts/init_porter_schema.py

python scripts/test_db.py
python scripts/check_db.py
python scripts/reset_lead.py
```
