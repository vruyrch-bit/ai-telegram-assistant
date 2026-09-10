# AI Telegram Assistant

Personal assistant upgrades are implemented locally. See [UPGRADE_NOTES.md](UPGRADE_NOTES.md) for features, configuration, validation and deployment steps.

A production-style multimodal AI assistant built with Python and deployed on Railway.

It combines conversational AI, PostgreSQL memory, natural-language task tools, voice transcription, document RAG, OCR, image understanding, a FastAPI admin API, Docker, automated tests, and GitHub Actions CI/CD.

## Features

- AI chat with persistent conversation memory
- Natural-language task creation, listing, completion, and deletion
- Telegram voice-message transcription
- PDF, DOCX, and TXT document support
- Hybrid semantic + keyword RAG
- OCR fallback for scanned PDF pages
- Photo, screenshot, and image understanding
- Image OCR for reading visible text
- Follow-up questions about the latest uploaded image
- FastAPI admin API with API-key authentication
- PostgreSQL persistence
- Dockerized deployment
- Railway production deployment
- Automated tests with pytest
- GitHub Actions CI

## Example Interactions

### Tasks

```text
Add finish calculus homework to my tasks
What tasks do I have?
Mark finish calculus homework as done
Delete finish calculus homework from my tasks
```

The assistant uses real tool calls for task operations and does not claim success unless the database action actually succeeds.

### Documents

Upload a PDF, DOCX, or TXT file and ask:

```text
Summarize this document
What are the main technical requirements?
Explain this section
Find the part about PostgreSQL
```

### Images

Send a photo or screenshot. The bot stores it and waits for instructions.

```text
User: [uploads image]

Bot:
Image received. Tell me what you'd like me to do with it.

User:
Read the text in this screenshot
```

You can also ask:

```text
Explain this diagram
What objects are visible?
What does this image show?
```

### Voice

Voice messages are transcribed and routed through the same assistant pipeline, so they can also trigger task tools and normal assistant actions.

## Architecture

```text
                         Telegram User
                               |
                               v
                    +----------------------+
                    |    Telegram Bot      |
                    |  Python / AsyncIO    |
                    +----------+-----------+
                               |
          +--------------------+----------------------+
          |                    |                      |
          v                    v                      v
      Text / Tasks         Voice Input          Files / Images
          |                    |                      |
          v                    v                      |
      LLM + Tools           Whisper                   |
          |                                           |
          |                         +-----------------+----------------+
          |                         |                                  |
          |                         v                                  v
          |                  Document Pipeline                  Vision Pipeline
          |                         |                                  |
          |                  PDF / DOCX / TXT                   Image + OCR
          |                         |
          |                  OCR when needed
          |                         |
          |                      FastEmbed
          |                         |
          |                  Hybrid Retrieval
          |                         |
          +-------------------------+-------------------------------+
                                    |
                                    v
                               PostgreSQL
                    +--------------------------------+
                    | Messages                       |
                    | Tasks                          |
                    | Documents                      |
                    | Document chunks                |
                    | Embeddings                     |
                    | Latest image                   |
                    +--------------------------------+

                           FastAPI Admin API
                                    |
                                    v
                               PostgreSQL
```

## Tech Stack

### Backend
- Python 3.12
- AsyncIO
- python-telegram-bot
- FastAPI
- Uvicorn

### AI
- Groq API
- GPT-OSS for chat and tool use
- Whisper for voice transcription
- Multimodal vision model
- Function/tool calling

### RAG and Documents
- FastEmbed
- `BAAI/bge-small-en-v1.5`
- NumPy cosine similarity
- Hybrid semantic + lexical retrieval
- PyPDF
- python-docx
- PyMuPDF
- Pillow
- Tesseract OCR

### Data and Infrastructure
- PostgreSQL
- psycopg
- Docker
- Docker Compose
- Railway
- GitHub Actions
- pytest
- pytest-asyncio

## Project Structure

```text
ai-telegram-assistant/
├── main.py
├── api.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── README.md
├── tests/
│   ├── test_api.py
│   └── test_core.py
└── .github/
    └── workflows/
        └── ci.yml
```

## RAG Pipeline

When a document is uploaded:

```text
Document
   |
   v
Text extraction
   |
   +--> OCR fallback for scanned PDF pages
   |
   v
Chunking
   |
   v
FastEmbed embeddings
   |
   v
PostgreSQL
```

When the user asks a document question:

```text
Question
   |
   v
Query embedding
   |
   +--> Semantic similarity
   |
   +--> Keyword relevance
   |
   v
Hybrid ranking
   |
   v
Best matching chunks
   |
   v
LLM answer
```

The retrieval score currently combines:

```text
85% semantic similarity
15% keyword relevance
```

## OCR

For PDFs, the bot first attempts normal text extraction. If a page contains too little readable text, the page is rendered with PyMuPDF and passed through Tesseract OCR.

OCR is page-by-page, so mixed PDFs containing both normal text and scanned pages are supported.

## Image Understanding

Supported image formats include:

- Telegram photos
- JPG / JPEG
- PNG
- WEBP
- Screenshots
- Diagrams
- Images containing text

The bot stores the latest uploaded image and waits for a follow-up instruction instead of immediately generating a long description.

Clear the latest stored image with:

```text
/clearimage
```

## Task Tool Calling

The assistant has persistent task tools:

- `create_task`
- `list_tasks`
- `complete_task`
- `delete_task`

Task data is stored in PostgreSQL.

## FastAPI Admin API

The project also exposes a separate FastAPI service.

Endpoints include:

```text
GET    /
GET    /health
GET    /stats
GET    /users
GET    /users/{telegram_user_id}/messages
DELETE /users/{telegram_user_id}/messages
```

Protected routes require the `X-API-Key` header.

## Environment Variables

Create a local `.env` file:

```env
TELEGRAM_TOKEN=your_telegram_token
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=your_postgresql_connection_string
ADMIN_API_KEY=your_admin_api_key
```

Never commit `.env` or real API keys to Git.

## Run Locally

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure PostgreSQL and Tesseract OCR are installed.

Start the Telegram bot:

```bash
python main.py
```

Start the API:

```bash
uvicorn api:app --reload
```

## Docker

Build the image:

```bash
docker build -t ai-telegram-assistant .
```

Run with Docker Compose:

```bash
docker compose up --build
```

## Tests

Run:

```bash
pytest -q
```

The test suite covers core behavior such as:

- FastAPI routes
- API authentication
- Telegram text cleanup
- Document chunking
- Keyword scoring
- Vector similarity
- Image-processing helpers
- Image follow-up detection
- Task-tool execution

## CI/CD

GitHub Actions runs on pushes and pull requests.

```text
Checkout
   |
   v
Python 3.12
   |
   v
Install dependencies
   |
   v
pip check
   |
   v
Compile Python
   |
   v
pytest
   |
   v
Docker build
```

## Production Deployment

The application is deployed on Railway using separate services:

```text
Telegram Bot
FastAPI API
PostgreSQL
```

The API includes a database-backed health endpoint:

```text
GET /health
```

## Security

The project includes several security measures:

- Secrets are loaded from environment variables
- `.env` is excluded from Git and Docker
- Admin API routes require an API key
- Uploaded document content is treated as untrusted data
- User data is separated by Telegram user ID
- Task actions are verified through actual tool results
- File and image size limits are enforced
- OCR work is capped
- Images are resized before vision processing

## What I Learned

This project was built to go beyond a basic chatbot wrapper and practice production-style AI engineering.

It involved:

- asynchronous Python
- third-party AI APIs
- REST API design
- authentication
- PostgreSQL
- database migrations
- AI tool calling
- embeddings
- retrieval-augmented generation
- hybrid search
- OCR
- multimodal AI
- file processing
- Docker
- cloud deployment
- automated testing
- CI/CD
- production debugging
- structured logging

## Status

The original portfolio baseline is complete. See UPGRADE_NOTES.md for the current personal-upgrade status and remaining work.

```text
[✓] AI chat
[✓] Persistent memory
[✓] Voice transcription
[✓] Natural-language task tools
[✓] FastAPI REST API
[✓] PostgreSQL
[✓] PDF / DOCX / TXT support
[✓] Semantic RAG
[✓] OCR
[✓] Image understanding
[✓] Docker
[✓] Railway deployment
[✓] Automated tests
[✓] GitHub Actions CI
```

## Author

Vruyr Chakhmakhchyan
