# AI Telegram Assistant

A full-stack AI-powered Telegram assistant built with Python, FastAPI, PostgreSQL, Docker, and OpenAI's GPT-OSS model through Groq.

The project demonstrates AI API integration, asynchronous programming, persistent conversation memory, REST API development, authentication, containerization, and database management.

## Features

- AI-powered Telegram chatbot
- OpenAI GPT-OSS 20B model through Groq
- Fast AI responses
- Persistent conversation memory
- PostgreSQL database
- Async Python architecture
- FastAPI REST API
- API-key authentication
- User and conversation management endpoints
- Health monitoring endpoint
- Bot statistics endpoint
- Docker and Docker Compose support
- Long Telegram message handling
- AI API timeout and error handling
- `/clear` command for deleting conversation memory

## Architecture

```text
                    Telegram User
                          |
                          v
                  Telegram Bot API
                          |
                          v
                     Python Bot
                    /          \
                   v            v
          Groq / GPT-OSS    PostgreSQL
                                ^
                                |
                             FastAPI
                                |
                                v
                          REST API Client
```

Docker Compose runs the main backend services:

```text
+---------------------------------------+
|             Docker Compose            |
|                                       |
|   Telegram Bot                        |
|        |                              |
|        +----------> PostgreSQL        |
|        |               ^              |
|        v               |              |
|   Groq API          FastAPI            |
|                                       |
+---------------------------------------+
```

## Technology Stack

### Backend

- Python
- FastAPI
- python-telegram-bot
- Psycopg 3

### AI

- OpenAI GPT-OSS 20B
- Groq API

### Database

- PostgreSQL

### Infrastructure

- Docker
- Docker Compose

### Development

- Git
- GitHub
- Linux

## Project Structure

```text
ai-telegram-assistant/
|
├── main.py
├── api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
├── .dockerignore
└── README.md
```

### `main.py`

Runs the Telegram bot and handles:

- Telegram messages
- AI requests
- PostgreSQL conversation memory
- `/start`
- `/clear`
- API errors
- long responses

### `api.py`

Provides the FastAPI backend with protected administrative endpoints.

### PostgreSQL

Stores persistent conversation history for each Telegram user.

## REST API

FastAPI provides interactive documentation at:

```text
http://localhost:8000/docs
```

### Public Endpoints

#### Health Check

```http
GET /health
```

Checks whether the API and PostgreSQL database are available.

### Protected Endpoints

Protected endpoints require:

```http
X-API-Key: YOUR_ADMIN_API_KEY
```

#### Statistics

```http
GET /stats
```

Returns information such as:

- total messages
- total users
- user messages
- AI messages

#### Users

```http
GET /users
```

Returns users with stored conversation history.

#### User Conversation

```http
GET /users/{telegram_user_id}/messages
```

Returns conversation history for a specific Telegram user.

#### Delete User Conversation

```http
DELETE /users/{telegram_user_id}/messages
```

Deletes the stored conversation memory for a user.

## Conversation Memory

Each conversation is stored in PostgreSQL.

Example:

```text
User:
Remember that my favorite color is purple.

AI:
Okay.

--- Bot restarted ---

User:
What is my favorite color?

AI:
Your favorite color is purple.
```

Because the conversation is stored in PostgreSQL, memory survives application and container restarts.

Only a limited number of recent messages are sent back to the AI model to keep requests efficient while the complete history remains stored in the database.

## Telegram Commands

### `/start`

Starts the assistant.

### `/clear`

Deletes the current user's stored conversation memory.

## Environment Variables

Create a `.env` file locally.

Example:

```env
TELEGRAM_TOKEN=your_telegram_bot_token
GROQ_API_KEY=your_groq_api_key
ADMIN_API_KEY=your_admin_api_key

POSTGRES_USER=telegrambot
POSTGRES_PASSWORD=your_database_password
POSTGRES_DB=telegram_ai_bot

DATABASE_URL=postgresql://telegrambot:your_database_password@localhost:5432/telegram_ai_bot
```

The real `.env` file is intentionally excluded from Git.

Never commit API keys, passwords, or tokens to the repository.

## Running With Docker

Build the containers:

```bash
docker compose build
```

Start all services:

```bash
docker compose up
```

Or run them in the background:

```bash
docker compose up -d
```

Check running services:

```bash
docker compose ps
```

View bot logs:

```bash
docker compose logs -f bot
```

View API logs:

```bash
docker compose logs -f api
```

Stop the system:

```bash
docker compose down
```

The PostgreSQL Docker volume keeps database data persistent between container restarts.

## Running Without Docker

Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Telegram bot:

```bash
python main.py
```

Run FastAPI in another terminal:

```bash
fastapi dev api.py
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Security

The project includes several basic security practices:

- Secrets are stored in environment variables
- `.env` is excluded from Git
- Administrative API endpoints require an API key
- SQL queries use parameterized values
- Conversation data is stored in PostgreSQL rather than exposed publicly
- Docker images do not include the `.env` file

## Error Handling

The bot handles common external API problems such as:

- AI rate limits
- connection failures
- timeouts
- empty AI responses
- Telegram message length limits

Long AI responses are automatically split into multiple Telegram messages.

## Current Development Status

The core system is working:

- [x] Telegram bot
- [x] AI integration
- [x] GPT-OSS through Groq
- [x] Persistent PostgreSQL memory
- [x] Async database access
- [x] FastAPI REST API
- [x] API-key authentication
- [x] Docker
- [x] Docker Compose
- [x] Git / GitHub
- [ ] Automated testing
- [ ] GitHub Actions CI/CD
- [ ] Cloud deployment
- [ ] Logging and monitoring
- [ ] Redis caching
- [ ] Vector database / RAG

## Future Improvements

Planned improvements include:

- CI/CD with GitHub Actions
- automated unit and integration tests
- cloud deployment
- structured application logging
- Redis caching
- database connection pooling
- rate limiting
- user authentication
- document upload and RAG
- vector database integration
- AI tool/function calling
- monitoring and production observability

## Purpose

This project was built as a practical software engineering and AI learning project.

The goal is not only to call an AI model, but to understand how a real AI-backed system is structured:

```text
User
  |
API / Event
  |
Application Logic
  |
AI Model + Database
  |
Infrastructure
```

It demonstrates how AI can be integrated into a backend application using modern software engineering practices.

## Author

**Vruyr Chakhmakhchyan**

GitHub: `vruyrch-bit`
