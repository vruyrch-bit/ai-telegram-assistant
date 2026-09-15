# Live smoke checkpoint — 15 September 2026

Repository: `/home/vruyr/telegram-ai-bot`, branch `personal-upgrades`.

## Production

Railway Telegram Bot still runs `a4048e9cb07209c2f762f10f746baf2b38d229dd`, deployment
`09a98284-55be-4551-9be7-2a7598b070ff`: SUCCESS, one RUNNING instance.
API deployment `1a1b00c1-23fb-4f80-96f8-ada935fd4db8` and Postgres deployment
`dc211bd9-1848-45e1-9449-131a1d20799c` remain unchanged.

No commit, push, deployment, credential change, production migration, or local
Telegram polling process was performed during this continuation.

## Live results

- `/health`: HTTP 200, healthy, database connected. Authenticated `/activity` and
  `/stats`: HTTP 200. Activity output inspected contained aggregate metrics only.
- Actual DDGS web search completed. Telegram received official Python download,
  version status, and release links, including a result titled Python 3.14.7.
  Aggregate activity recorded one web_search call with zero failures (911 ms).
  Groq returned 429 during answer composition. The deployed recovery code correctly
  kept the sources and explained that it could not finish the answer. This is a
  pass for search execution and graceful recovery, but the complete natural-language
  answer remains limited by the provider quota.
- Memory replacement into a forgotten duplicate passed live. Saved tagged amber
  preference ID 11, then requested exact blue wording already present in forgotten
  fixture ID 10. `/memory` showed only the blue test preference and the existing
  user preference; the amber entry was no longer active.
- Memory replacement into an active duplicate passed live. Saved tagged rehearsal
  room fact ID 12, then requested replacement with the existing blue preference.
  `/memory` still showed one test preference, with no duplicate or rehearsal entry.
- Both replacements succeeded before the final AI response hit 429. Telegram
  correctly reported `Memory updated. Check /memory.` Aggregate activity recorded
  two replace_memory calls with zero failures (average 1177 ms).
- `/forget 1` removed the positively identified test preference. Final `/memory`
  shows the preserved `I prefer concise answers`, numbered 1. Aggregate active
  memory count returned to 1. Forgotten fixture rows and chat history remain.
- `/files` still shows both pre-existing `AI Engineer internship.pdf` documents.
  No document was uploaded, and `/clearfiles` was not used.
- No new tasks, notes, images, or reminders were created during this continuation.
  Image live tests had already passed on 14 September and were not repeated.
- Final `/reminders`: `No upcoming reminders.`

Retrieved deployment logs showed the original startup Conflict only:

```text
2026-09-14T09:47:04.862839405Z Telegram handler failed error=Conflict
```

An earlier network error appeared before today's tests; its cause is not established:

```text
2026-09-15T01:11:31.312312958Z Telegram handler failed error=NetworkError
```

Today's commands and successful tool activity demonstrate recovery. Rate-limit
warnings appeared at 08:06:34, 08:09:25, and 08:13:56 UTC during the web and memory
tests (`AI request failed error=RateLimitError status=429`). No database, migration, or
reminder-worker failures were observed in the retrieved logs. Log inspection and
Railway instance state are not an independent container process audit.

## Local changes awaiting review and deployment

An existing uncommitted OpenRouter draft was found in `config.py`,
`services/ai_requests.py`, `services/cloud_fallback.py`, and
`tests/test_ai_reliability.py`. Its routing and tests were preserved. This continuation
fixed the adapter and added regression coverage:

- Vision previously failed before reaching OpenRouter with
  `TypeError: request_openrouter_completion() got an unexpected keyword argument 'temperature'`.
  The adapter now accepts the actual vision options, forwards images and temperature,
  translates the token budget, and requests excluded reasoning using OpenRouter's
  supported format. Groq-specific reasoning options are not forwarded unchanged.
- OpenRouter fallback accepts only `openrouter/free` or a `:free` model. Requests
  additionally cap prompt, completion, request, and image pricing at zero. Providers
  must support the requested parameters. Missing keys or a non-free configuration
  keep this fallback inactive. Free model capacity is not guaranteed.
- Empty/malformed fallback responses produce the existing friendly error. Secondary
  failures are bounded and sanitized; cancellation propagates. A successful primary
  response never contacts the fallback. Tests verify that completed tool actions
  are not replayed when the final completion switches providers.
- Test setup clears any inherited OpenRouter key, keeping all tests off real services.

Neither the local environment nor the Railway bot had an OpenRouter key at inspection
time. No real request was sent to OpenRouter. Activating it would send fallback
conversation context, tool results, and images when applicable to OpenRouter and its
selected model provider. Ollama remains disabled.

The document test exposed a missing cleanup path: production can clear all documents,
but cannot remove one. Added locally:

- `/files` shows stable database IDs and uses long-message splitting.
- `/deletefile ID` previews the named file; `/deletefile ID confirm` deletes that
  owned document and its searchable chunks using the existing cascade constraint.
- User ownership is checked for preview and deletion. Invalid, missing, or other
  users' IDs do not delete data or disclose their filenames.
- Help, startup text, and command registration include the new command.

Changed/added files: `config.py`, `services/ai_requests.py`,
`services/cloud_fallback.py`, `bot/app.py`, `bot/commands.py`,
`bot/personal_commands.py`, `database/documents.py`, `tests/conftest.py`,
`tests/test_ai_reliability.py`, `tests/test_cloud_fallback.py`,
`tests/test_document_cleanup.py`, and this checkpoint.

## Validation

- Initial existing draft tests: 27 passed.
- Focused fallback, document cleanup, and feature flow tests: 66 passed.
- Full final suite with disposable PostgreSQL and cached offline Whisper:
  **163 passed**, four existing dependency deprecation warnings.
- Compilation, `git diff --check`, and installed dependency consistency passed.
- XML results: `/home/vruyr/Documents/ChatGPT/test/bot-validation-2026-09-15.xml`.
- Document deletion tests preserve two same-named original documents and a separate
  user's document, and check confirmation, ownership, chunk cleanup, and registration.

## Resume here

1. Review the local diff. The new changes are uncommitted and undeployed. The user's
   earlier production constraint requires approval for another bot-only rollout.
2. Obtain/configure an OpenRouter key if the user wants the secondary cloud fallback
   active; use the free route. Do not print keys or put them in a commit. The key can
   be created through https://openrouter.ai/settings/keys.
3. After approval, commit and push the intended changes. Pushing this branch triggers
   Railway's bot auto-deployment. Preserve API, Postgres, credentials, and settings.
4. After the old bot instance is gone, live-test fallback and a small uniquely named
   TXT document: upload, retrieval, document-specific/collection search, and removal
   through the new single-document command. Preserve both original PDFs.
5. Remaining older checklist items include complete recurring-task coverage and a
   fully composed web answer. Do not call earlier partial checks a complete pass.
6. Investigate the previously observed pre-deployment `?` message that led to a stale
   smoke-test reminder before declaring every natural-language action reliable. It
   was not reproduced or fixed during this continuation.

Provider references used for the adapter:
- https://openrouter.ai/docs/guides/routing/routers/free-router
- https://openrouter.ai/docs/guides/routing/provider-selection
- https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
