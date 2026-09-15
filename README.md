# AI Telegram Assistant

A production-style personal AI assistant built with Python and Telegram.

The project combines conversational AI, persistent PostgreSQL storage, long-term memory, document retrieval, image understanding, web search, task management, reminders, and multi-provider AI fallback in one Telegram interface.

It was built as a practical AI engineering project focused on reliability, tool use, retrieval, persistence, deployment, and safe failure handling.

---

## Features

### AI Chat
- Natural-language conversations through Telegram
- Groq as the primary cloud inference provider
- OpenRouter free-model fallback when the primary provider is unavailable or rate-limited
- Provider stickiness during multi-step tool workflows
- Bounded retries and user-friendly failure messages
- Reasoning output is filtered from user-visible responses

### Long-Term Memory
- Persistent user memories stored in PostgreSQL
- Semantic memory retrieval
- Memory deduplication
- Preference replacement
- Importance and recency ranking
- Explicit memory commands:
  - `/remember`
  - `/memory`
  - `/forget`

### Document RAG
Supports:

- PDF
- scanned PDF with OCR
- TXT
- DOCX

Documents are:

1. parsed,
2. split into chunks,
3. embedded,
4. stored,
5. retrieved semantically when relevant.

Features include:

- hybrid semantic + keyword retrieval
- document collections
- document-specific searching
- stable document IDs
- ownership validation
- single-document deletion with confirmation

Commands:

```text
/files
/deletefile ID
```

### Image Understanding
- Upload photos and screenshots
- OCR text extraction
- Vision-model analysis
- Persistent recent image history
- Query the latest uploaded image naturally
- Query older images by ID
- Clear stored image history

Commands:

```text
/images
/image ID your question
/clearimages confirm
```

### Web Search
- Current public web search through DDGS
- No separate search API key required
- Search results are passed back to the AI for answer composition
- Completed search results are preserved if the AI provider fails afterward

Example:

```text
Search the web for the latest Python release.
```

### Task Management
Tasks support:

- creation
- completion
- deletion
- editing
- priority
- project
- notes
- deadlines
- search
- daily recurrence
- weekly recurrence

Task numbers shown to the user are clean `1, 2, 3...` values.

Internal PostgreSQL primary keys remain private and are not exposed to the user.

Example:

```text
Create a high priority task to order Sitka spruce tomorrow.
Complete task 1.
Make task 2 high priority.
```

Command:

```text
/tasks
```

### Recurring Tasks

Completing a recurring task automatically creates its next occurrence while preserving:

- title
- priority
- project
- notes
- recurrence settings

### Reminders
- Natural-language reminder creation
- User timezone support
- One-time reminders
- Daily reminders
- Weekly reminders
- Persistent PostgreSQL scheduling

Examples:

```text
Remind me tomorrow at 5 PM.
Remind me every day at 8 PM.
```

Commands:

```text
/reminders
/timezone Asia/Yerevan
```

### Personal Notes
Persistent project and knowledge notes with search support.

Example:

```text
Save a project note that the guitar top should use solid Sitka spruce.
```

Command:

```text
/notes
```

### Voice
- Telegram voice-message transcription
- Whisper-based speech recognition

### Admin API
A separate FastAPI service provides authenticated administrative endpoints for:

- health status
- application statistics
- activity information

Example:

```text
GET /health
GET /stats
GET /activity
```

Protected endpoints require an API key.

---

## Architecture

```text
Telegram
   │
   ▼
python-telegram-bot
   │
   ▼
AI Orchestrator
   │
   ├── Groq
   │      │
   │      └── OpenRouter fallback
   │
   ├── Task tools
   ├── Memory tools
   ├── Reminder tools
   ├── Notes
   ├── Web search
   ├── Vision / OCR
   └── Document RAG
             │
             ▼
         PostgreSQL
```

Main project structure:

```text
telegram-ai-bot/
├── main.py
├── api.py
├── config.py
│
├── bot/
│   ├── app.py
│   ├── commands.py
│   ├── handlers.py
│   ├── document_handler.py
│   └── personal_commands.py
│
├── database/
│   ├── core.py
│   ├── memory.py
│   ├── tasks.py
│   ├── documents.py
│   ├── images.py
│   ├── long_term_memory.py
│   ├── personal.py
│   └── upgrades.py
│
├── services/
│   ├── ai.py
│   ├── ai_requests.py
│   ├── ai_responses.py
│   ├── cloud_fallback.py
│   ├── local_ai.py
│   ├── voice.py
│   ├── vision.py
│   ├── ocr.py
│   ├── document_parser.py
│   ├── memory.py
│   ├── jobs.py
│   ├── scheduling.py
│   └── web_search.py
│
├── rag/
│   ├── embeddings.py
│   └── retrieval.py
│
├── tools/
│   ├── task_tools.py
│   ├── memory_tools.py
│   └── personal_tools.py
│
├── utils/
│   └── telegram_text.py
│
├── scripts/
│   └── test_disposable_postgres.py
│
└── tests/
```

---

## Tech Stack

### Backend
- Python
- asyncio
- FastAPI
- python-telegram-bot

### Database
- PostgreSQL
- psycopg

### AI
- Groq
- OpenRouter
- optional Ollama adapter
- Whisper speech recognition
- vision-capable language models

### Retrieval
- FastEmbed
- `BAAI/bge-small-en-v1.5`
- hybrid semantic and keyword retrieval

### Documents
- PyMuPDF
- python-docx
- Tesseract OCR
- Pillow

### Search
- DDGS

### Infrastructure
- Docker
- Railway
- GitHub
- GitHub Actions

---

## AI Provider Reliability

The assistant uses a fallback strategy:

```text
Groq
  │
  ├── success → continue
  │
  └── temporary failure / rate limit
          │
          ▼
     OpenRouter
          │
          ▼
   remain on OpenRouter
   for the rest of the
   current tool workflow
```

Provider stickiness is important for multi-step tool calls.

For example:

```text
AI requests web search
→ web search runs
→ primary provider becomes rate-limited
→ OpenRouter takes over
→ remaining tool conversation stays on OpenRouter
→ final answer is generated
```

This prevents tool-call histories from being passed back and forth between incompatible providers during the same request.

---

## Environment Variables

Create a `.env` file locally or configure equivalent variables in your deployment platform.

Example:

```env
TELEGRAM_BOT_TOKEN=your_telegram_token

DATABASE_URL=postgresql://...

GROQ_API_KEY=your_groq_key

OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=openrouter/free

ADMIN_API_KEY=your_admin_api_key
```

Additional configuration variables may be available in `config.py`.

Never commit real API keys or credentials to Git.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/vruyrch-bit/ai-telegram-assistant.git
cd ai-telegram-assistant
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure environment variables, then run:

```bash
python main.py
```

---

## Running the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Health endpoint:

```text
GET /health
```

---

## Testing

Standard test suite:

```bash
pytest -q
```

Dependency verification:

```bash
python -m pip check
```

Whitespace verification:

```bash
git diff --check
```

Compile verification:

```bash
python -m compileall -q \
  main.py \
  api.py \
  config.py \
  database \
  rag \
  services \
  tools \
  bot \
  utils \
  tests \
  scripts
```

The project also includes a disposable PostgreSQL validation script:

```bash
BOT_TEST_LOCAL_WHISPER=1 \
HF_HUB_OFFLINE=1 \
python scripts/test_disposable_postgres.py -q
```

Current full disposable-database validation:

```text
164 passed
```

---

## Deployment

The project is deployed using Railway with separate services for:

```text
Telegram Bot
FastAPI API
PostgreSQL
```

PostgreSQL provides persistent state across deployments.

Docker configuration is also included for reproducible deployment.

---

## Reliability and Safety

The project includes several protections intended for production-style behavior:

- private-chat access controls
- optional user-ID allowlist
- request rate limiting
- database ownership checks
- bounded AI retries
- cloud-provider fallback
- protection against accidental paid OpenRouter model routing
- safe error messages
- sanitized logging
- persistent tool-result handling
- memory deduplication
- transaction locking for sensitive updates
- document ownership validation
- confirmation before destructive document deletion
- hidden internal database task IDs
- reasoning-output filtering

Secrets are never intentionally included in logs or bot responses.

---

## Current Status

Core functionality is implemented and deployed.

Validated areas include:

- AI chat
- PostgreSQL persistence
- document RAG
- scanned-document OCR
- image understanding
- image history
- long-term memory
- task management
- recurring tasks
- reminders
- notes
- DDGS web search
- Groq → OpenRouter fallback
- multi-provider tool workflows
- FastAPI administration API
- Railway deployment
- production smoke testing

---

## Possible Future Improvements

Potential future additions include:

- local Ollama inference
- spoken AI responses
- streamed Telegram responses
- web dashboard
- usage quotas
- configurable data-retention policies
- additional model providers
- more advanced agent workflows
- richer observability and analytics

---

## Why I Built This

This project was built to move beyond a simple chatbot and explore the engineering required for a persistent AI application.

It covers several concepts important to practical AI engineering:

- LLM tool calling
- retrieval-augmented generation
- embeddings
- semantic search
- persistent memory
- multimodal AI
- OCR
- asynchronous Python
- structured validation
- PostgreSQL
- APIs
- containerization
- cloud deployment
- automated testing
- provider failover
- reliability engineering

The goal was to build an assistant that can not only generate text, but also interact with persistent tools and data while continuing to behave predictably when external AI services fail.

---

## Author

**Vruyr Chakhmakhchyan**

GitHub: [@vruyrch-bit](https://github.com/vruyrch-bit)
