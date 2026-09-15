# Reliability fixes — September 14, 2026

These changes follow live failures on deployed commit `36c1f61685ea2898fcfaf17deef613ccb382da5f`.

## Changes

- Memory replacement now handles text that matches another active or forgotten memory. It reuses the matching record and deactivates the superseded record in one transaction. Saves and replacements for the same user are serialized to prevent a concurrent insert from recreating the uniqueness error. The returned database ID can change when a duplicate is reused. No rows are deleted and no schema migration or bulk data update is included.
- Vision requests for the configured Groq Qwen 3.6/3.8 models request a final answer without reasoning. Replies also filter provider reasoning markers, handle empty output, and disclose output truncation. Old reasoning blocks are filtered from assistant history when constructing new requests; stored chat history is not rewritten.
- Text and vision completions retry once for temporary failures when the server's requested delay is at most five seconds. Longer cooldowns return a friendly error without retrying early. Automatic SDK retries are disabled for these two clients so retries do not stack. Configuration/authentication errors are not retried. This does not remove the provider's usage limits.
- If an AI response fails after tools have completed, the bot reports successful mutations and preserves web source links. It does not falsely report failed actions as successful or replay application tools as part of a completion retry. Invalid model-generated tool calls produce a friendly response, using the installed Groq SDK's actual error-body shape.
- DDGS selects from its available engines automatically instead of requesting the unavailable Bing backend. No search API key or paid fallback is added.
- Changed text/voice/tool failure logs record error classes instead of full provider or database exception payloads, which can contain private text or arguments.

## Validation

- Focused checks: 88 passed, one opt-in local Whisper check skipped in that run.
- Full suite with disposable local PostgreSQL and cached offline Whisper: **139 passed**, four existing dependency deprecation warnings. No production database was used by the tests.
- Compilation, whitespace checks, and installed dependency consistency passed.
- A direct live Groq request using the existing harmless blue-circle fixture returned `The image shows a blue circle.` without `<think>` output. This verifies the new local request code against Groq; it is not a post-deployment Telegram test.
- A direct live DDGS call returned five results, including official Python download and version-status pages, without the old backend warning. This is not a complete Telegram question/answer test.
- Existing live API checks: health 200; unauthenticated activity 401; authenticated activity and stats 200. Activity returned aggregate metric fields only.

## Deployment boundary and remaining work

The changes are prepared locally. At validation time Railway still ran `36c1f61` in one Telegram Bot instance. The API and Postgres deployments were unchanged. No production rows, credentials, environment variables, or service configuration were modified during this validation.

Pushing to the tracked `personal-upgrades` branch automatically deploys the Telegram Bot. Review/approval of this new bot-only rollout is needed under the user's existing deployment constraint. After deployment, verify memory replacement into both active and forgotten duplicates, latest/older image answers, graceful rate-limit responses, and logs. Use tagged disposable fixtures and preserve existing user data and the two original PDFs.

A second cloud-provider fallback is **not implemented or configured**. No new provider credentials are needed for these fixes. Choosing and enabling a separate cloud fallback remains a later step; Ollama has not been enabled. Telegram live verification of this new code remains pending deployment.

References: [Groq reasoning controls](https://console.groq.com/docs/reasoning), [Groq rate-limit behavior](https://console.groq.com/docs/rate-limits), [DDGS backend selection](https://github.com/deedy5/ddgs).
