# AI Telegram Assistant

A cloud-deployed AI Telegram assistant built with Python, FastAPI, PostgreSQL, Docker, GitHub Actions, and OpenAI's GPT-OSS 20B model through Groq.

The project demonstrates how an AI model can be integrated into a real backend system with persistent memory, REST APIs, authentication, automated testing, CI, containerization, cloud deployment, and structured logging.

[![CI](https://github.com/vruyrch-bit/ai-telegram-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/vruyrch-bit/ai-telegram-assistant/actions/workflows/ci.yml)

---

## Live API

FastAPI is deployed on Railway:

**API**
```text
https://api-product-863.up.railway.app
```

**Health Check**
```text
https://api-product-863.up.railway.app/health
```

**Interactive API Documentation**
```text
https://api-product-863.up.railway.app/docs
```

---

## Features

- AI-powered Telegram chatbot
- OpenAI GPT-OSS 20B through Groq
- Persistent conversation memory
- PostgreSQL database
- Async Python architecture
- FastAPI REST API
- API-key authentication
- User and conversation administration endpoints
- Health monitoring endpoint
- Usage statistics
- Docker containerization
- Docker Compose local environment
- Railway cloud deployment
- GitHub Actions CI
- Automated API tests with Pytest
- Structured application logging
- AI timeout and rate-limit handling
- Telegram long-message handling
- `/clear` command for deleting user memory
- Secrets managed through environment variables

---

# Architecture

```text
                         ┌─────────────────┐
                         │ Telegram User   │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Telegram Bot API│
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   Python Bot    │
                         │     Async       │
                         └───────┬─────────┘
                                 │
                     ┌───────────┴───────────┐
                     │                       │
                     ▼                       ▼
              ┌─────────────┐         ┌──────────────┐
              │ Groq API    │         │ PostgreSQL   │
              │ GPT-OSS 20B │         │ Conversation │
              └─────────────┘         │ Memory       │
                                      └──────▲───────┘
                                             │
                                             │
                                      ┌──────┴───────┐
                                      │   FastAPI    │
                                      │   REST API   │
                                      └──────┬───────┘
                                             │
                                             ▼
                                      API / Swagger UI
```

---

# Cloud Architecture

The application is deployed using Railway.

```text
Railway Project

├── Telegram Bot
│     ├── Python
│     ├── Telegram API
│     └── Groq / GPT-OSS
│
├── FastAPI
│     ├── REST endpoints
│     ├── API authentication
│     └── Public HTTPS domain
│
└── PostgreSQL
      └── Persistent conversation storage
```

The Telegram bot and FastAPI service share the same PostgreSQL database.

---

# Technology Stack

## Backend

- Python 3.12
- FastAPI
- python-telegram-bot
- Psycopg 3

## AI

- OpenAI GPT-OSS 20B
- Groq API

## Database

- PostgreSQL

## Infrastructure

- Docker
- Docker Compose
- Railway

## Testing / CI

- Pytest
- GitHub Actions
- Docker build verification

## Development

- Git
- GitHub
- Linux

---

# Project Structure

```text
ai-telegram-assistant/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── tests/
│   └── test_api.py
│
├── main.py
├── api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── .gitignore
├── .dockerignore
└── README.md
```

---

# Telegram Bot

The Telegram service is responsible for:

- receiving Telegram messages
- loading previous conversation history
- sending messages to the AI model
- saving conversations to PostgreSQL
- handling API failures
- splitting long Telegram responses
- clearing user memory
- structured production logging

Available commands:

```text
/start
```

Starts the assistant.

```text
/clear
```

Deletes the user's stored conversation history.

---

# Persistent Conversation Memory

Conversation history is stored in PostgreSQL.

Example:

```text
User:
Remember that my favorite color is purple.

Assistant:
Okay.

--- Application restarted ---

User:
What is my favorite color?

Assistant:
Your favorite color is purple.
```

The memory survives:

- Python restarts
- Docker container restarts
- Railway redeployments

The complete history is stored in PostgreSQL, while only a limited number of recent messages are provided to the AI model for each request.

---

# REST API

The FastAPI backend provides management and monitoring endpoints.

Interactive Swagger documentation:

```text
https://api-product-863.up.railway.app/docs
```

---

## Public Endpoints

### Root

```http
GET /
```

Returns basic service information.

---

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "database": "connected"
}
```

This endpoint verifies that the API can successfully communicate with PostgreSQL.

---

# Protected Endpoints

Administrative endpoints require:

```http
X-API-Key: YOUR_ADMIN_API_KEY
```

---

## Statistics

```http
GET /stats
```

Returns statistics including:

- total messages
- total users
- user messages
- AI messages

---

## Users

```http
GET /users
```

Returns Telegram users that have stored conversation history.

---

## User Conversation

```http
GET /users/{telegram_user_id}/messages
```

Optional query parameter:

```text
limit=20
```

Returns recent stored messages for a Telegram user.

---

## Delete Conversation

```http
DELETE /users/{telegram_user_id}/messages
```

Deletes the stored conversation for that user.

---

# Authentication

Administrative endpoints use API-key authentication.

Requests must provide:

```http
X-API-Key: YOUR_ADMIN_API_KEY
```

Requests with a missing or invalid key receive:

```http
401 Unauthorized
```

Secrets are never stored directly in the source code.

---

# Environment Variables

The application uses environment variables for configuration.

Example `.env`:

```env
TELEGRAM_TOKEN=your_telegram_token

GROQ_API_KEY=your_groq_api_key

ADMIN_API_KEY=your_admin_api_key

POSTGRES_USER=telegrambot

POSTGRES_PASSWORD=your_database_password

POSTGRES_DB=telegram_ai_bot

DATABASE_URL=postgresql://telegrambot:your_database_password@localhost:5432/telegram_ai_bot
```

The real `.env` file is excluded from Git.

Never commit:

- API keys
- Telegram bot tokens
- database passwords
- administrative secrets

---

# Running Locally

Clone the repository:

```bash
git clone https://github.com/vruyrch-bit/ai-telegram-assistant.git
```

Enter the project:

```bash
cd ai-telegram-assistant
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your `.env` file.

Then run the Telegram bot:

```bash
python main.py
```

Run FastAPI in another terminal:

```bash
fastapi dev api.py
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

# Running With Docker

Build the services:

```bash
docker compose build
```

Start the system:

```bash
docker compose up
```

Run in detached mode:

```bash
docker compose up -d
```

Check services:

```bash
docker compose ps
```

View Telegram bot logs:

```bash
docker compose logs -f bot
```

View FastAPI logs:

```bash
docker compose logs -f api
```

Stop all containers:

```bash
docker compose down
```

PostgreSQL data is stored in a Docker volume and persists between normal container restarts.

---

# Automated Testing

The project uses Pytest.

Run tests locally:

```bash
pytest -q
```

Current automated tests verify behavior including:

- root API endpoint
- authentication requirement
- protected `/stats` endpoint
- protected `/users` endpoint
- rejection of invalid API keys

---

# CI Pipeline

GitHub Actions automatically runs when code is pushed to `main` or when a pull request targets `main`.

The CI pipeline performs:

```text
Checkout source
       ↓
Set up Python 3.12
       ↓
Install dependencies
       ↓
Check dependency consistency
       ↓
Check Python syntax
       ↓
Run automated tests
       ↓
Build Docker image
       ↓
Pass ✅ / Fail ❌
```

This helps catch problems before relying on a new deployment.

---

# Cloud Deployment

The project is deployed on Railway.

Production services:

```text
Telegram Bot      ✅
FastAPI           ✅
PostgreSQL        ✅
```

Railway provides:

- container deployment
- environment variable management
- PostgreSQL hosting
- persistent storage
- HTTPS networking
- deployment logs
- GitHub integration

The bot can continue running even when the development computer is turned off.

---

# Logging and Monitoring

The application uses structured logging instead of relying only on basic `print()` statements.

Example Telegram logs:

```text
INFO | telegram-bot | Message received user_id=...
INFO | telegram-bot | AI response received user_id=...
INFO | telegram-bot | Conversation saved user_id=...
```

Example API logs:

```text
INFO | api | GET /health -> 200 | 18.42ms
INFO | api | GET /stats -> 401 | 2.14ms
```

User message contents are intentionally not included in standard application logs.

The `/health` endpoint can also be used for service monitoring.

---

# Error Handling

The bot handles several common failures:

- AI API rate limits
- AI timeouts
- connection errors
- empty AI responses
- Telegram message size limits
- PostgreSQL connection errors
- invalid API authentication

Long AI responses are automatically divided into Telegram-compatible message sizes.

---

# Security

The project includes several basic security practices:

- secrets stored in environment variables
- `.env` excluded from Git
- API-key protected administrative endpoints
- parameterized PostgreSQL queries
- no user conversation text in normal logs
- `.env` excluded from Docker images
- database not exposed directly to public users
- protected conversation-management endpoints

---

# Development Workflow

```text
Developer
    │
    ▼
Code change
    │
    ▼
Git commit
    │
    ▼
GitHub push
    │
    ▼
GitHub Actions
    │
    ├── Tests
    ├── Syntax checks
    └── Docker build
    │
    ▼
Railway deployment
    │
    ▼
Production
```

---

# What I Learned

This project helped me gain practical experience with:

- asynchronous Python
- event-driven applications
- REST API design
- AI model integration
- prompt and conversation management
- PostgreSQL
- persistent application state
- authentication
- Docker
- cloud deployment
- Git workflows
- CI pipelines
- automated testing
- debugging external APIs
- structured production logging
- environment-based configuration

The goal of the project was not simply to call an AI model, but to understand how an AI-powered application can be structured as a real backend system.

---

# Future Improvements

Possible future additions include:

- Redis caching
- database connection pooling
- application-level rate limiting
- user authentication
- document upload
- Retrieval-Augmented Generation (RAG)
- vector database support
- AI tool/function calling
- advanced monitoring
- Kubernetes deployment
- Terraform infrastructure
- more integration and load tests

---

# Status

### Version 1

- [x] Telegram bot
- [x] GPT-OSS AI integration
- [x] PostgreSQL persistent memory
- [x] Async processing
- [x] FastAPI REST API
- [x] API authentication
- [x] Docker
- [x] Docker Compose
- [x] Railway cloud deployment
- [x] Git / GitHub
- [x] GitHub Actions CI
- [x] Automated tests
- [x] Docker build validation
- [x] Structured logging
- [x] Health monitoring
- [x] Public API documentation

---

# Author

**Vruyr Chakhmakhchyan**

GitHub: [vruyrch-bit](https://github.com/vruyrch-bit)
