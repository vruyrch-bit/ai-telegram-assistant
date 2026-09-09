AI Telegram Assistant — Multimodal AI Engineering Project

I built and deployed a production-style AI Telegram assistant as a portfolio project focused on backend and AI engineering.

The project started as a simple Telegram chatbot and gradually developed into a full multimodal assistant with persistent data, AI tools, RAG, OCR, voice processing, image understanding, an admin API, testing, CI/CD, and cloud deployment.

Main features
-AI chat with persistent PostgreSQL conversation memory
-Natural-language task management using AI tool/function calling
-Voice-message transcription with Whisper
-PDF, DOCX, and TXT processing
-Semantic RAG using FastEmbed and BAAI/bge-small-en-v1.5
-Hybrid semantic + keyword document retrieval
-OCR for scanned PDFs using Tesseract and PyMuPDF
-Photo, screenshot, and diagram understanding
-OCR for text inside images
-Follow-up questions about uploaded images
-FastAPI REST API with API-key authentication
-PostgreSQL database
-Docker containerization
-Railway deployment
-Automated testing with pytest
-GitHub Actions CI/CD
-Structured production logging

   Architecture

Telegram
   ↓
Async Python Bot
   ├── AI Chat / Tool Calling
   ├── Whisper → Voice
   ├── FastEmbed → RAG
   ├── Tesseract → OCR
   ├── Vision AI → Images
   └── PostgreSQL → Persistent Data

FastAPI Admin API
   ↓
PostgreSQL

One of the parts I found most interesting was building the document retrieval pipeline.

Instead of only searching documents for matching words, the bot creates embeddings for document chunks and combines:

85% semantic similarity
15% keyword relevance

This allows it to answer questions even when the user's wording is different from the wording in the source document.

I also implemented OCR fallback page-by-page: normal PDF text extraction is attempted first, and Tesseract is only used when a page appears to be scanned.

Technologies

Python • AsyncIO • FastAPI • PostgreSQL • Docker • Railway • GitHub Actions • Groq • GPT-OSS • Whisper • FastEmbed • Tesseract OCR • PyMuPDF • pytest

What I learned

This project gave me hands-on experience with asynchronous Python, external AI APIs, REST APIs, authentication, PostgreSQL, AI tool calling, embeddings, RAG, OCR, multimodal models, Docker, CI/CD, production deployment, automated testing, and debugging a real deployed application.

The project is now feature-complete and will be part of my AI/backend engineering portfolio.

Repository:
https://github.com/vruyrch-bit/ai-telegram-assistant
