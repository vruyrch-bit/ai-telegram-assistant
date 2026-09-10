import os

from dotenv import load_dotenv


# ==================================================
# MODELS
# ==================================================

AI_MODEL = "openai/gpt-oss-20b"
VOICE_MODEL = "whisper-large-v3-turbo"
VISION_MODEL = "qwen/qwen3.6-27b"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384


# ==================================================
# MEMORY / TOOL SETTINGS
# ==================================================

MEMORY_MESSAGE_LIMIT = 30
MAX_TOOL_ROUNDS = 5


# ==================================================
# FILE SIZE LIMITS
# ==================================================

MAX_VOICE_SIZE = 20 * 1024 * 1024
MAX_DOCUMENT_SIZE = 20 * 1024 * 1024
MAX_IMAGE_SIZE = 15 * 1024 * 1024


# ==================================================
# RAG SETTINGS
# ==================================================

DOCUMENT_CHUNK_SIZE = 1400
DOCUMENT_CHUNK_OVERLAP = 200
MAX_DOCUMENT_CONTEXT = 12000
DOCUMENT_RETRIEVAL_LIMIT = 6
SUMMARY_CHUNK_LIMIT = 8

SEMANTIC_WEIGHT = 0.85
KEYWORD_WEIGHT = 0.15
MIN_SEMANTIC_SCORE = 0.38


# ==================================================
# OCR SETTINGS
# ==================================================

MIN_PAGE_TEXT_CHARS = 40
OCR_SCALE = 2.0
MAX_OCR_PAGES = 30
OCR_LANGUAGE = "eng"


# ==================================================
# IMAGE SETTINGS
# ==================================================

MAX_IMAGE_DIMENSION = 2048
MAX_IMAGE_OCR_CONTEXT = 6000


# ==================================================
# ENVIRONMENT
# ==================================================

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
FASTEMBED_CACHE_DIR = os.getenv("FASTEMBED_CACHE_DIR")


if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN was not found")

AI_PROVIDER = os.getenv("AI_PROVIDER", "groq")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
if AI_PROVIDER not in {"groq", "local"}:
    raise ValueError("AI_PROVIDER must be groq or local")

if AI_PROVIDER == "groq" and not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY was not found")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL was not found")


# Optional upgrades; existing model defaults remain usable.
AI_MODEL = os.getenv("AI_MODEL", AI_MODEL)
VOICE_MODEL = os.getenv("VOICE_MODEL", VOICE_MODEL)
VISION_MODEL = os.getenv("VISION_MODEL", VISION_MODEL)
CODING_MODEL = os.getenv("CODING_MODEL") or AI_MODEL
REASONING_MODEL = os.getenv("REASONING_MODEL") or AI_MODEL
ALLOWED_USER_IDS = {int(value.strip()) for value in os.getenv("ALLOWED_USER_IDS", "").split(",") if value.strip()}

VOICE_LANGUAGE = os.getenv("VOICE_LANGUAGE", "")

if AI_PROVIDER == "local":
    # Local model names never fall through to paid cloud providers.
    AI_MODEL = os.getenv("LOCAL_AI_MODEL", "qwen3:4b")
    CODING_MODEL = AI_MODEL
    REASONING_MODEL = AI_MODEL
    VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "qwen3-vl:2b")
    VOICE_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "base")
