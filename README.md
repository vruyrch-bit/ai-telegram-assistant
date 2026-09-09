# AI Telegram Assistant

A production-style multimodal AI assistant built with Python and deployed on Railway.

The project combines conversational AI, persistent memory, voice transcription, document processing, semantic RAG, OCR, image understanding, AI tool calling, PostgreSQL, FastAPI, Docker, automated tests, and CI/CD.

## Features

### AI Chat
- Asynchronous Telegram chatbot
- Groq-hosted LLM integration
- Persistent conversation memory
- Plain-text Telegram-friendly responses
- Long-message splitting
- Structured error handling and logging

### AI Task Management
The assistant can manage persistent tasks through natural language.

Examples:

- `Add finish calculus homework to my tasks`
- `What tasks do I have?`
- `Mark finish calculus homework as done`
- `Delete finish calculus homework from my tasks`

The AI uses function/tool calling instead of pretending an action succeeded.

Commands are also available:

- `/tasks`
- `/addtask`
- `/donetask`
- `/deletetask`

### Voice Messages
Users can send Telegram voice messages.

Pipeline:

```text
Telegram voice message
        ↓
Groq Whisper transcription
        ↓
Normal assistant pipeline
        ↓
Memory / RAG / tools
        ↓
Telegram response
