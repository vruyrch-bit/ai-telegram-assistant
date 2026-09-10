# Personal assistant upgrades — September 10, 2026

## Implemented locally

- Memory ranking keeps the existing relevance threshold and importance bonus, adds a maximum 0.05 recency bonus (30-day half-life) and 0.03 recent-use bonus (7-day half-life). Only selected memories get access timestamps; failed timestamp writes do not discard results.
- Durable reminders: create/list/edit/cancel, explicit saved IANA timezone, one-time/daily/weekly schedules, five-second polling, retry backoff, terminal failures after five attempts. `/timezone`, `/reminders` and normal-language tools are connected.
- Tasks: search/edit priority, project, notes and deadlines; daily/weekly recurring tasks create their next occurrence on completion. Recurring tasks require YYYY-MM-DD deadlines; deadlines alone do not send notifications.
- Personal knowledge: create/edit/search/delete notes and goals grouped by project. `/notes` searches this store.
- Web search: DDGS provides free public search without an API key. Live testing returned five sources for a Python release query, but some attempts were temporarily unavailable. Malformed result links are discarded safely; there is no paid fallback.
- Documents: collection assignment, explicit document/collection search tools, source labels with document IDs and chunk numbers, Unicode keyword extraction, rejection of irrelevant tail chunks.
- Images: last ten uploads retained per user; `/images`, `/image ID question`, `/clearimages confirm`. Existing `/clearimage` still clears only the latest-image slot.
- Model routing: configurable coding and reasoning models selected by simple request keywords; default to the existing chat model. This is basic rule-based routing, not a quality-trained classifier.
- Voice: downloaded-size checks, long transcript handling, optional transcription language hint. Local faster-whisper uses its bundled speech detector to avoid transcribing silence. Groq transcription remains available. Voice replies are not implemented.
- Telegram: `/help` command menu and buttons, global error handler, private-chat enforcement, optional user allowlist, 20 updates/minute per-user limit. Excess requests are silently ignored. The in-memory limiter resets on restart.
- Observability: tool name/success/duration events without arguments or message bodies; authenticated `/activity` aggregates recent tools, reminder status and memory counts.
- Reliability: repeatable additive schema migrations, bounded new database connections/statements, stronger API-key comparison, backup utility, and CI now runs both unit and disposable PostgreSQL integration tests on main and personal-upgrades pushes.

## Configuration

Existing secrets are unchanged. Add optional settings in your deployment environment:

- Web search needs no additional key.
- `AI_PROVIDER=local`, `OLLAMA_URL=http://127.0.0.1:11434`, `LOCAL_AI_MODEL=qwen3:4b`, `LOCAL_VISION_MODEL=qwen3-vl:2b`: optional PC-only setup requiring a running Ollama server, downloaded models, and `requirements-local.txt` for voice. The local Ollama installation is not yet complete. The standard Docker image installs only `requirements.txt`, so local Whisper is not included there.
- `AI_MODEL`, `CODING_MODEL`, `REASONING_MODEL`, `VISION_MODEL`, `VOICE_MODEL`: model identifiers supported by your existing Groq account. Chat/coding/reasoning choices must support tool calls. Alternate models were not live-tested.
- `VOICE_LANGUAGE`: optional transcription language code; omit for automatic detection.
- `ALLOWED_USER_IDS`: comma-separated Telegram numeric user IDs. Empty allows any private-chat user, preserving existing access availability. Group chats are now ignored to avoid sharing personal context.

Timezone is deliberately not inferred: each user sets `/timezone` or explicitly states their timezone before creating reminders.

## Deployment and verification

Code has been changed locally, not pushed or deployed. Database tests used a separate temporary PostgreSQL cluster. Production data and secrets were not changed, and no real Telegram notifications were sent.

Before deploying, back up your production database. The migrations run during bot startup and add tables/columns without dropping existing data. Keep exactly one polling bot instance. The reminder worker starts/stops with that bot, so the bot service must remain running for delivery.

The API `/activity` endpoint requires the bot migration to have run; before then it reports unavailable. Redeploy both bot and API to expose all changes.

After deployment: set timezone; create a reminder a few minutes ahead; verify receipt; test a recurring task, note search and older-image lookup. Web/provider checks require their real configuration.

## Delivery and scale limits

- Reminder delivery is at-least-once. A crash between successful Telegram delivery and database commit can duplicate a notification. Cancellation cannot retract a message already being delivered.
- Missed recurring reminders are coalesced into one delivery and advanced to the next future occurrence. Spring-forward nonexistent times shift forward; subsequent recurrences use the shifted time. Recurring task dates use the server calendar (typically UTC).
- Image history has a count bound; documents, notes, reminders and activity logs do not yet have automatic retention quotas. Logs contain metadata; schedule cleanup appropriate to your usage.
- Memory retrieval still uses a 100-candidate window and per-memory reads. Document retrieval uses at most 500 chunks. No pgvector migration or empirical relevance evaluation has been added.
- No arbitrary shell/code execution tools, autonomous web monitoring, scheduled AI summaries, synthesized voice replies, streaming chat, or visual admin dashboard are implemented. Those remain follow-up work rather than being presented as finished.

## Backup and restore

With `DATABASE_URL` already set in the shell environment and PostgreSQL client tools installed:

```bash
python scripts/backup_database.py /secure/location/bot-backup.dump
```

The script refuses overwrite and creates mode-0600 output. Keep backups outside Git, encrypt off-site copies and establish a retention schedule. A failed backup file is incomplete and must not be used.

Test restores into a separate empty database using `pg_restore --exit-on-error`; supply credentials through a private PostgreSQL service file or environment variables, not a shell command containing a password. Never point a restore test at production.

## Tests

Latest local validation: 100 tests passed with disposable PostgreSQL and cached local Whisper enabled. The ordinary suite passes 71 tests and skips 29 opt-in tests. Compilation, dependency checks, Docker build/import checks, Compose validation, workflow YAML/shell syntax, and backup/restore passed. Four dependency deprecation warnings remain in the full run. Real Telegram polling and live Groq/Ollama inference were not exercised; AI responses, vision, OCR and embeddings use deterministic fixtures in integration tests.

```bash
python -m compileall -q main.py api.py config.py bot database services rag tools utils scripts tests
python -m pytest -q
python -m pip check
python scripts/test_disposable_postgres.py -q
# Only when faster-whisper and its base model are already installed/cached:
BOT_TEST_LOCAL_WHISPER=1 HF_HUB_OFFLINE=1 python scripts/test_disposable_postgres.py -q
```

The disposable runner creates a fresh UTF-8 PostgreSQL cluster with a private Unix socket and removes it after testing. It requires local PostgreSQL server/client tools. Alternatively, set `BOT_TEST_DATABASE_URL` to an explicitly disposable local database whose name starts with `bot_test`. The integration fixture truncates application tables between tests; never use any database containing real data. CI provisions its own PostgreSQL service. Tests disable dotenv loading and use dummy application credentials.

## Technical references

- [PostgreSQL queue locking](https://www.postgresql.org/docs/18/sql-select.html)
- [Telegram application lifecycle](https://docs.python-telegram-bot.org/en/v22.6/telegram.ext.applicationbuilder.html)
