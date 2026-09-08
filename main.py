kimport io
import logging
import os
import re
import sys

import groq
import psycopg

from dotenv import load_dotenv
from groq import AsyncGroq

from docx import Document
from pypdf import PdfReader

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


# ==================================================
# LOGGING
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "telegram-bot | "
        "%(name)s | "
        "%(message)s"
    ),
    stream=sys.stdout,
    force=True,
)

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ==================================================
# SETTINGS
# ==================================================

AI_MODEL = "openai/gpt-oss-20b"

VOICE_MODEL = "whisper-large-v3-turbo"

MEMORY_MESSAGE_LIMIT = 30

MAX_VOICE_SIZE = 20 * 1024 * 1024

MAX_DOCUMENT_SIZE = 20 * 1024 * 1024

DOCUMENT_CHUNK_SIZE = 3000

DOCUMENT_CHUNK_OVERLAP = 300

MAX_DOCUMENT_CONTEXT = 12000


# ==================================================
# ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")


if not TELEGRAM_TOKEN:
    raise ValueError(
        "TELEGRAM_TOKEN was not found"
    )

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY was not found"
    )

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL was not found"
    )


# ==================================================
# GROQ CLIENT
# ==================================================

client = AsyncGroq(
    api_key=GROQ_API_KEY,
    timeout=20.0,
    max_retries=1,
)


# ==================================================
# DATABASE INITIALIZATION
# ==================================================

async def initialize_database():

    logger.info(
        "Connecting to PostgreSQL"
    )

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        # ------------------------------------------
        # Conversation memory
        # ------------------------------------------

        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id BIGSERIAL PRIMARY KEY,

                telegram_user_id BIGINT NOT NULL,

                role VARCHAR(20) NOT NULL,

                content TEXT NOT NULL,

                created_at TIMESTAMPTZ
                    DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_messages_user

            ON messages (
                telegram_user_id,
                id
            )
            """
        )


        # ------------------------------------------
        # Documents
        # ------------------------------------------

        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id BIGSERIAL PRIMARY KEY,

                telegram_user_id BIGINT NOT NULL,

                filename TEXT NOT NULL,

                file_type VARCHAR(20) NOT NULL,

                created_at TIMESTAMPTZ
                    DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


        # ------------------------------------------
        # Document chunks
        # ------------------------------------------

        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id BIGSERIAL PRIMARY KEY,

                document_id BIGINT NOT NULL
                    REFERENCES documents(id)
                    ON DELETE CASCADE,

                chunk_index INTEGER NOT NULL,

                content TEXT NOT NULL
            )
            """
        )

        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_documents_user

            ON documents (
                telegram_user_id,
                id
            )
            """
        )

        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_document_chunks_document

            ON document_chunks (
                document_id,
                chunk_index
            )
            """
        )


        # ------------------------------------------
        # Tasks
        # ------------------------------------------

        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id BIGSERIAL PRIMARY KEY,

                telegram_user_id BIGINT NOT NULL,

                title TEXT NOT NULL,

                status VARCHAR(20)
                    NOT NULL
                    DEFAULT 'open',

                due_date TEXT,

                created_at TIMESTAMPTZ
                    DEFAULT CURRENT_TIMESTAMP,

                completed_at TIMESTAMPTZ
            )
            """
        )

        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_tasks_user

            ON tasks (
                telegram_user_id,
                id
            )
            """
        )

    logger.info(
        "PostgreSQL database ready"
    )


# ==================================================
# SAVE CONVERSATION MESSAGE
# ==================================================

async def save_message(
    telegram_user_id: int,
    role: str,
    content: str,
):

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        await connection.execute(
            """
            INSERT INTO messages (
                telegram_user_id,
                role,
                content
            )

            VALUES (%s, %s, %s)
            """,
            (
                telegram_user_id,
                role,
                content,
            ),
        )


# ==================================================
# LOAD CONVERSATION MEMORY
# ==================================================

async def load_memory(
    telegram_user_id: int,
):

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        cursor = await connection.execute(
            """
            SELECT
                role,
                content

            FROM messages

            WHERE telegram_user_id = %s

            ORDER BY id DESC

            LIMIT %s
            """,
            (
                telegram_user_id,
                MEMORY_MESSAGE_LIMIT,
            ),
        )

        rows = await cursor.fetchall()

    rows.reverse()

    messages = []

    for role, content in rows:

        if role == "model":
            role = "assistant"

        if role not in (
            "user",
            "assistant",
            "system",
        ):
            continue

        messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    return messages


# ==================================================
# DELETE MEMORY
# ==================================================

async def delete_memory(
    telegram_user_id: int,
):

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        await connection.execute(
            """
            DELETE FROM messages

            WHERE telegram_user_id = %s
            """,
            (
                telegram_user_id,
            ),
        )


# ==================================================
# TASK DATABASE FUNCTIONS
# ==================================================

async def create_task(
    telegram_user_id: int,
    title: str,
    due_date=None,
):

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        cursor = await connection.execute(
            """
            INSERT INTO tasks (
                telegram_user_id,
                title,
                due_date
            )

            VALUES (%s, %s, %s)

            RETURNING id
            """,
            (
                telegram_user_id,
                title,
                due_date,
            ),
        )

        row = await cursor.fetchone()

    return row[0]


async def get_tasks(
    telegram_user_id: int,
):

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        cursor = await connection.execute(
            """
            SELECT
                id,
                title,
                status,
                due_date

            FROM tasks

            WHERE telegram_user_id = %s

            ORDER BY
                CASE
                    WHEN status = 'open'
                    THEN 0
                    ELSE 1
                END,

                id ASC
            """,
            (
                telegram_user_id,
            ),
        )

        return await cursor.fetchall()


async def complete_task(
    telegram_user_id: int,
    task_id: int,
):

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        cursor = await connection.execute(
            """
            UPDATE tasks

            SET
                status = 'done',

                completed_at =
                    CURRENT_TIMESTAMP

            WHERE
                id = %s

                AND telegram_user_id = %s

                AND status = 'open'

            RETURNING title
            """,
            (
                task_id,
                telegram_user_id,
            ),
        )

        row = await cursor.fetchone()

    if not row:
        return None

    return row[0]


async def delete_task(
    telegram_user_id: int,
    task_id: int,
):

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        cursor = await connection.execute(
            """
            DELETE FROM tasks

            WHERE
                id = %s

                AND telegram_user_id = %s

            RETURNING title
            """,
            (
                task_id,
                telegram_user_id,
            ),
        )

        row = await cursor.fetchone()

    if not row:
        return None

    return row[0]


# ==================================================
# PDF TEXT EXTRACTION
# ==================================================

def extract_pdf_text(
    file_bytes: bytes,
):

    reader = PdfReader(
        io.BytesIO(file_bytes)
    )

    text_parts = []

    for page in reader.pages:

        page_text = (
            page.extract_text()
            or ""
        )

        if page_text.strip():

            text_parts.append(
                page_text
            )

    return "\n\n".join(
        text_parts
    )


# ==================================================
# DOCX TEXT EXTRACTION
# ==================================================

def extract_docx_text(
    file_bytes: bytes,
):

    document = Document(
        io.BytesIO(file_bytes)
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        text = (
            paragraph.text.strip()
        )

        if text:

            paragraphs.append(
                text
            )

    return "\n".join(
        paragraphs
    )


# ==================================================
# TXT TEXT EXTRACTION
# ==================================================

def extract_txt_text(
    file_bytes: bytes,
):

    try:

        return file_bytes.decode(
            "utf-8"
        )

    except UnicodeDecodeError:

        return file_bytes.decode(
            "latin-1",
            errors="ignore",
        )


# ==================================================
# DOCUMENT CHUNKING
# ==================================================

def chunk_text(
    text: str,
):

    text = text.strip()

    chunks = []

    start = 0

    while start < len(text):

        end = (
            start
            + DOCUMENT_CHUNK_SIZE
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        if end >= len(text):
            break

        start = (
            end
            - DOCUMENT_CHUNK_OVERLAP
        )

    return chunks


# ==================================================
# SAVE DOCUMENT
# ==================================================

async def save_document(
    telegram_user_id: int,
    filename: str,
    file_type: str,
    chunks,
):

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        cursor = await connection.execute(
            """
            INSERT INTO documents (
                telegram_user_id,
                filename,
                file_type
            )

            VALUES (%s, %s, %s)

            RETURNING id
            """,
            (
                telegram_user_id,
                filename,
                file_type,
            ),
        )

        row = await cursor.fetchone()

        document_id = row[0]

        for index, chunk in enumerate(
            chunks
        ):

            await connection.execute(
                """
                INSERT INTO document_chunks (
                    document_id,
                    chunk_index,
                    content
                )

                VALUES (%s, %s, %s)
                """,
                (
                    document_id,
                    index,
                    chunk,
                ),
            )

    return document_id


# ==================================================
# LOAD DOCUMENTS
# ==================================================

async def load_documents(
    telegram_user_id: int,
):

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        cursor = await connection.execute(
            """
            SELECT
                id,
                filename,
                file_type,
                created_at

            FROM documents

            WHERE telegram_user_id = %s

            ORDER BY id DESC

            LIMIT 20
            """,
            (
                telegram_user_id,
            ),
        )

        return await cursor.fetchall()


# ==================================================
# DELETE DOCUMENTS
# ==================================================

async def delete_documents(
    telegram_user_id: int,
):

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        await connection.execute(
            """
            DELETE FROM documents

            WHERE telegram_user_id = %s
            """,
            (
                telegram_user_id,
            ),
        )


# ==================================================
# DOCUMENT SEARCH
# ==================================================

STOP_WORDS = {
    "the",
    "and",
    "that",
    "this",
    "what",
    "where",
    "when",
    "with",
    "from",
    "have",
    "does",
    "about",
    "into",
    "your",
    "would",
    "could",
    "should",
    "there",
    "they",
    "them",
    "then",
    "than",
    "are",
    "was",
    "were",
    "for",
    "how",
    "who",
    "why",
}


def question_words(
    question: str,
):

    words = re.findall(
        r"[A-Za-z0-9]+",
        question.lower(),
    )

    return {
        word
        for word in words
        if (
            len(word) >= 3
            and word not in STOP_WORDS
        )
    }


# ==================================================
# GET DOCUMENT CONTEXT
# ==================================================

async def get_document_context(
    telegram_user_id: int,
    question: str,
):

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        cursor = await connection.execute(
            """
            SELECT
                d.id,
                d.filename,
                dc.chunk_index,
                dc.content

            FROM document_chunks dc

            JOIN documents d
                ON d.id = dc.document_id

            WHERE d.telegram_user_id = %s

            ORDER BY
                d.id DESC,
                dc.chunk_index ASC

            LIMIT 300
            """,
            (
                telegram_user_id,
            ),
        )

        rows = await cursor.fetchall()

    if not rows:

        return None

    q_words = question_words(
        question
    )

    lower_question = (
        question.lower()
    )

    summary_request = any(
        phrase in lower_question
        for phrase in (
            "summarize",
            "summary",
            "summarise",
            "this file",
            "this document",
            "the document",
            "the file",
            "pdf",
        )
    )

    scored = []

    newest_document_id = (
        rows[0][0]
    )

    for (
        document_id,
        filename,
        chunk_index,
        content,
    ) in rows:

        lower_content = (
            content.lower()
        )

        score = sum(
            lower_content.count(
                word
            )
            for word in q_words
        )

        scored.append(
            (
                score,
                document_id,
                filename,
                chunk_index,
                content,
            )
        )

    if summary_request:

        selected = [
            item
            for item in scored
            if item[1]
            == newest_document_id
        ][:4]

    else:

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        selected = [
            item
            for item in scored
            if item[0] > 0
        ][:4]

    if not selected:

        return None

    context_parts = []

    total_length = 0

    for (
        score,
        document_id,
        filename,
        chunk_index,
        content,
    ) in selected:

        section = (
            f"[Source: {filename}, "
            f"chunk {chunk_index}]\n"
            f"{content}"
        )

        if (
            total_length
            + len(section)
            > MAX_DOCUMENT_CONTEXT
        ):
            break

        context_parts.append(
            section
        )

        total_length += len(
            section
        )

    if not context_parts:

        return None

    return "\n\n".join(
        context_parts
    )


# ==================================================
# CLEAN TELEGRAM OUTPUT
# ==================================================

def clean_telegram_text(
    text: str,
):

    text = text.replace(
        "**",
        ""
    )

    text = text.replace(
        "__",
        ""
    )

    text = text.replace(
        "```",
        ""
    )

    text = text.replace(
        "`",
        ""
    )

    lines = []

    for line in text.splitlines():

        line = re.sub(
            r"^\s*#{1,6}\s*",
            "",
            line,
        )

        line = re.sub(
            r"^\s*[-*]\s+",
            "• ",
            line,
        )

        lines.append(
            line
        )

    cleaned = "\n".join(
        lines
    )

    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )

    return cleaned.strip()


# ==================================================
# /start
# ==================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    await update.message.reply_text(
        "Hello! 👋\n\n"

        "I am an AI assistant powered by "
        "OpenAI GPT-OSS 20B through Groq.\n\n"

        "You can send me:\n"
        "• Text messages 💬\n"
        "• Voice messages 🎤\n"
        "• PDF files 📄\n"
        "• TXT files 📝\n"
        "• DOCX files 📘\n\n"

        "Task commands:\n"
        "/tasks - show your tasks\n"
        "/addtask - add a task\n"
        "/donetask - complete a task\n"
        "/deletetask - delete a task\n\n"

        "Other commands:\n"
        "/clear - clear conversation memory\n"
        "/files - show uploaded files\n"
        "/clearfiles - delete uploaded files"
    )


# ==================================================
# /clear
# ==================================================

async def clear_memory(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    telegram_user_id = (
        update.effective_user.id
    )

    await delete_memory(
        telegram_user_id
    )

    logger.info(
        "Memory cleared user_id=%s",
        telegram_user_id,
    )

    await update.message.reply_text(
        "Conversation memory cleared. 🧹"
    )


# ==================================================
# /files
# ==================================================

async def files_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    telegram_user_id = (
        update.effective_user.id
    )

    documents = await load_documents(
        telegram_user_id
    )

    if not documents:

        await update.message.reply_text(
            "You haven't uploaded any files yet."
        )

        return

    lines = [
        "📁 Your uploaded files:"
    ]

    for (
        document_id,
        filename,
        file_type,
        created_at,
    ) in documents:

        lines.append(
            f"• {filename}"
        )

    await update.message.reply_text(
        "\n".join(lines)
    )


# ==================================================
# /clearfiles
# ==================================================

async def clear_files(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    telegram_user_id = (
        update.effective_user.id
    )

    await delete_documents(
        telegram_user_id
    )

    logger.info(
        "Documents cleared user_id=%s",
        telegram_user_id,
    )

    await update.message.reply_text(
        "Uploaded documents deleted. 🗑️"
    )


# ==================================================
# /tasks
# ==================================================

async def tasks_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    telegram_user_id = (
        update.effective_user.id
    )

    tasks = await get_tasks(
        telegram_user_id
    )

    if not tasks:

        await update.message.reply_text(
            "You don't have any tasks yet."
        )

        return

    lines = [
        "✅ Your Tasks",
        "",
    ]

    for (
        task_id,
        title,
        status,
        due_date,
    ) in tasks:

        if status == "done":
            icon = "✅"
        else:
            icon = "⬜"

        line = (
            f"{icon} {task_id}. {title}"
        )

        if due_date:

            line += (
                f" — due {due_date}"
            )

        lines.append(
            line
        )

    await update.message.reply_text(
        "\n".join(lines)
    )


# ==================================================
# /addtask
# ==================================================

async def add_task_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    telegram_user_id = (
        update.effective_user.id
    )

    if not context.args:

        await update.message.reply_text(
            "Usage:\n"
            "/addtask Finish calculus homework"
        )

        return

    title = " ".join(
        context.args
    ).strip()

    task_id = await create_task(
        telegram_user_id,
        title,
    )

    logger.info(
        "Task created user_id=%s task_id=%s",
        telegram_user_id,
        task_id,
    )

    await update.message.reply_text(
        f"✅ Task added.\n\n"
        f"{task_id}. {title}"
    )


# ==================================================
# /donetask
# ==================================================

async def done_task_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    telegram_user_id = (
        update.effective_user.id
    )

    if not context.args:

        await update.message.reply_text(
            "Usage:\n"
            "/donetask 3"
        )

        return

    try:

        task_id = int(
            context.args[0]
        )

    except ValueError:

        await update.message.reply_text(
            "Task ID must be a number."
        )

        return

    title = await complete_task(
        telegram_user_id,
        task_id,
    )

    if not title:

        await update.message.reply_text(
            "I couldn't find that open task."
        )

        return

    logger.info(
        "Task completed user_id=%s task_id=%s",
        telegram_user_id,
        task_id,
    )

    await update.message.reply_text(
        f"✅ Completed:\n{title}"
    )


# ==================================================
# /deletetask
# ==================================================

async def delete_task_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    telegram_user_id = (
        update.effective_user.id
    )

    if not context.args:

        await update.message.reply_text(
            "Usage:\n"
            "/deletetask 3"
        )

        return

    try:

        task_id = int(
            context.args[0]
        )

    except ValueError:

        await update.message.reply_text(
            "Task ID must be a number."
        )

        return

    title = await delete_task(
        telegram_user_id,
        task_id,
    )

    if not title:

        await update.message.reply_text(
            "I couldn't find that task."
        )

        return

    logger.info(
        "Task deleted user_id=%s task_id=%s",
        telegram_user_id,
        task_id,
    )

    await update.message.reply_text(
        f"🗑️ Deleted:\n{title}"
    )


# ==================================================
# SEND LONG TELEGRAM MESSAGE
# ==================================================

async def send_long_message(
    update: Update,
    text: str,
):

    text = clean_telegram_text(
        text
    )

    max_length = 4000

    for i in range(
        0,
        len(text),
        max_length,
    ):

        part = text[
            i:i + max_length
        ]

        await update.message.reply_text(
            part
        )


# ==================================================
# ASK AI
# ==================================================

async def ask_ai(
    user_message: str,
    history,
    document_context=None,
):

    messages = [
        {
            "role": "system",

            "content": (
                "You are an AI assistant running "
                "inside a custom Telegram bot. "

                "You are powered by OpenAI's "
                "GPT-OSS 20B model through Groq. "

                "You are not ChatGPT and must not "
                "claim to be ChatGPT. "

                "Give clear, accurate, and useful answers. "

                "Use plain text only. "

                "Do not use Markdown tables. "

                "Do not use Markdown formatting symbols "
                "such as **, __, #, or backticks. "

                "Use simple headings, numbered sections, "
                "and bullet points beginning with •. "

                "Keep paragraphs short and readable "
                "on a phone screen. "

                "When document context is provided, "
                "use it as the primary source for "
                "questions about the uploaded document. "

                "Do not invent facts that are not "
                "supported by the document context. "

                "If the provided document context is "
                "not sufficient, say so clearly."
            ),
        }
    ]

    messages.extend(
        history
    )

    if document_context:

        messages.append(
            {
                "role": "system",

                "content": (
                    "Relevant uploaded-document "
                    "context follows:\n\n"
                    + document_context
                ),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    logger.info(
        "Sending request model=%s "
        "document_context=%s",
        AI_MODEL,
        bool(document_context),
    )

    response = (
        await client.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            reasoning_effort="low",
            max_completion_tokens=1500,
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ==================================================
# PROCESS USER MESSAGE
# ==================================================

async def process_user_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_message: str,
):

    telegram_user_id = (
        update.effective_user.id
    )

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    history = await load_memory(
        telegram_user_id
    )

    document_context = (
        await get_document_context(
            telegram_user_id,
            user_message,
        )
    )

    answer = await ask_ai(
        user_message,
        history,
        document_context,
    )

    if not answer:

        await update.message.reply_text(
            "The AI returned an empty response."
        )

        return

    answer = clean_telegram_text(
        answer
    )

    await save_message(
        telegram_user_id,
        "user",
        user_message,
    )

    await save_message(
        telegram_user_id,
        "assistant",
        answer,
    )

    logger.info(
        "Conversation saved user_id=%s",
        telegram_user_id,
    )

    await send_long_message(
        update,
        answer,
    )


# ==================================================
# TEXT MESSAGE
# ==================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    telegram_user_id = (
        update.effective_user.id
    )

    logger.info(
        "Text message received user_id=%s",
        telegram_user_id,
    )

    try:

        await process_user_message(
            update,
            context,
            update.message.text,
        )

    except groq.RateLimitError:

        await update.message.reply_text(
            "The AI rate limit has been reached. "
            "Please try again shortly."
        )

    except groq.APITimeoutError:

        await update.message.reply_text(
            "The AI took too long to respond."
        )

    except groq.APIConnectionError:

        logger.exception(
            "Groq connection error "
            "user_id=%s",
            telegram_user_id,
        )

        await update.message.reply_text(
            "I couldn't connect to the AI service."
        )

    except Exception:

        logger.exception(
            "Unexpected text-message error "
            "user_id=%s",
            telegram_user_id,
        )

        await update.message.reply_text(
            "Something went wrong."
        )


# ==================================================
# VOICE TRANSCRIPTION
# ==================================================

async def transcribe_voice(
    audio_bytes: bytes,
):

    logger.info(
        "Sending voice message to Whisper"
    )

    transcription = (
        await client.audio.transcriptions.create(
            file=(
                "voice.ogg",
                audio_bytes,
            ),

            model=VOICE_MODEL,

            response_format="json",

            temperature=0.0,
        )
    )

    return transcription.text


# ==================================================
# VOICE MESSAGE
# ==================================================

async def handle_voice(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    telegram_user_id = (
        update.effective_user.id
    )

    voice = (
        update.message.voice
    )

    logger.info(
        "Voice received user_id=%s",
        telegram_user_id,
    )

    try:

        if (
            voice.file_size
            and voice.file_size
            > MAX_VOICE_SIZE
        ):

            await update.message.reply_text(
                "That voice message is too large."
            )

            return

        await update.message.reply_text(
            "🎤 Listening..."
        )

        telegram_file = (
            await context.bot.get_file(
                voice.file_id
            )
        )

        audio_data = (
            await telegram_file.download_as_bytearray()
        )

        transcription = (
            await transcribe_voice(
                bytes(audio_data)
            )
        ).strip()

        if not transcription:

            await update.message.reply_text(
                "I couldn't understand the "
                "voice message."
            )

            return

        await update.message.reply_text(
            f"📝 I heard:\n{transcription}"
        )

        await process_user_message(
            update,
            context,
            transcription,
        )

    except Exception:

        logger.exception(
            "Unexpected voice-message error "
            "user_id=%s",
            telegram_user_id,
        )

        await update.message.reply_text(
            "I couldn't process that voice message."
        )


# ==================================================
# DOCUMENT MESSAGE
# ==================================================

async def handle_document(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    telegram_user_id = (
        update.effective_user.id
    )

    document = (
        update.message.document
    )

    filename = (
        document.file_name
        or "document"
    )

    extension = (
        os.path.splitext(
            filename
        )[1]
        .lower()
    )

    logger.info(
        "Document received user_id=%s "
        "filename=%s",
        telegram_user_id,
        filename,
    )

    try:

        if (
            document.file_size
            and document.file_size
            > MAX_DOCUMENT_SIZE
        ):

            await update.message.reply_text(
                "That file is too large."
            )

            return

        if extension not in (
            ".pdf",
            ".txt",
            ".docx",
        ):

            await update.message.reply_text(
                "I currently support only "
                "PDF, TXT, and DOCX files."
            )

            return

        await update.message.reply_text(
            "📄 Reading your file..."
        )

        telegram_file = (
            await context.bot.get_file(
                document.file_id
            )
        )

        file_data = (
            await telegram_file.download_as_bytearray()
        )

        file_bytes = bytes(
            file_data
        )

        if extension == ".pdf":

            extracted_text = extract_pdf_text(
                file_bytes
            )

            file_type = "pdf"

        elif extension == ".docx":

            extracted_text = extract_docx_text(
                file_bytes
            )

            file_type = "docx"

        else:

            extracted_text = extract_txt_text(
                file_bytes
            )

            file_type = "txt"

        extracted_text = (
            extracted_text.strip()
        )

        if not extracted_text:

            await update.message.reply_text(
                "I couldn't extract readable text "
                "from this file.\n\n"

                "If it is a scanned PDF made from "
                "images, OCR support will be needed."
            )

            return

        chunks = chunk_text(
            extracted_text
        )

        await save_document(
            telegram_user_id,
            filename,
            file_type,
            chunks,
        )

        logger.info(
            "Document stored user_id=%s "
            "filename=%s chunks=%s",
            telegram_user_id,
            filename,
            len(chunks),
        )

        await update.message.reply_text(
            "✅ File processed successfully.\n\n"

            f"📄 File: {filename}\n"

            f"📚 Text chunks stored: "
            f"{len(chunks)}\n\n"

            "You can now ask me:\n\n"

            "• Summarize this document\n"
            "• What are the main points?\n"
            "• Explain a section\n"
            "• Find information inside the file"
        )

    except Exception:

        logger.exception(
            "Document processing error "
            "user_id=%s",
            telegram_user_id,
        )

        await update.message.reply_text(
            "I couldn't process that file."
        )


# ==================================================
# STARTUP
# ==================================================

async def post_init(
    application: Application,
):

    await initialize_database()

    logger.info(
        "Telegram bot initialization complete"
    )


# ==================================================
# RUN BOT
# ==================================================

def main():

    logger.info(
        "Starting AI Telegram Bot"
    )

    logger.info(
        "Text model=%s",
        AI_MODEL,
    )

    logger.info(
        "Voice model=%s",
        VOICE_MODEL,
    )

    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )


    # ----------------------------------------------
    # General commands
    # ----------------------------------------------

    application.add_handler(
        CommandHandler(
            "start",
            start,
        )
    )

    application.add_handler(
        CommandHandler(
            "clear",
            clear_memory,
        )
    )


    # ----------------------------------------------
    # Documents
    # ----------------------------------------------

    application.add_handler(
        CommandHandler(
            "files",
            files_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "clearfiles",
            clear_files,
        )
    )


    # ----------------------------------------------
    # Tasks
    # ----------------------------------------------

    application.add_handler(
        CommandHandler(
            "tasks",
            tasks_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "addtask",
            add_task_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "donetask",
            done_task_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "deletetask",
            delete_task_command,
        )
    )


    # ----------------------------------------------
    # Voice
    # ----------------------------------------------

    application.add_handler(
        MessageHandler(
            filters.VOICE,
            handle_voice,
        )
    )


    # ----------------------------------------------
    # Documents
    # ----------------------------------------------

    application.add_handler(
        MessageHandler(
            filters.Document.ALL,
            handle_document,
        )
    )


    # ----------------------------------------------
    # Text
    # ----------------------------------------------

    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            handle_message,
        )
    )


    logger.info(
        "Starting Telegram polling"
    )

    application.run_polling()


# ==================================================
# START
# ==================================================

if __name__ == "__main__":
    main()
