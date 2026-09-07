import logging
import os
import sys

import groq
import psycopg

from dotenv import load_dotenv
from groq import AsyncGroq

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

# Reduce noisy HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ==================================================
# SETTINGS
# ==================================================

AI_MODEL = "openai/gpt-oss-20b"

MEMORY_MESSAGE_LIMIT = 30


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

    logger.info(
        "PostgreSQL memory database ready"
    )


# ==================================================
# SAVE MESSAGE
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
# LOAD MEMORY
# ==================================================

async def load_memory(
    telegram_user_id: int,
):

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        cursor = await connection.execute(
            """
            SELECT role, content
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

        # Compatibility with old Gemini messages
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
        "I have persistent PostgreSQL memory.\n\n"
        "Send me anything you'd like to talk about.\n\n"
        "Use /clear to delete our conversation memory."
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
# SEND LONG TELEGRAM MESSAGE
# ==================================================

async def send_long_message(
    update: Update,
    text: str,
):

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
                "If asked who you are, explain that "
                "you are a custom AI Telegram "
                "assistant powered by GPT-OSS 20B. "
                "Give clear and useful answers. "
                "Keep answers reasonably concise "
                "unless the user asks for detail."
            ),
        }
    ]

    messages.extend(
        history
    )

    messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    logger.info(
        "Sending request model=%s",
        AI_MODEL,
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
# HANDLE USER MESSAGE
# ==================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    telegram_user_id = (
        update.effective_user.id
    )

    user_message = (
        update.message.text
    )

    # Do NOT log the actual user's message
    logger.info(
        "Message received user_id=%s",
        telegram_user_id,
    )

    try:

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING,
        )

        history = await load_memory(
            telegram_user_id
        )

        logger.info(
            "Loaded memory user_id=%s messages=%s",
            telegram_user_id,
            len(history),
        )

        answer = await ask_ai(
            user_message,
            history,
        )

        if not answer:

            logger.warning(
                "Empty AI response user_id=%s",
                telegram_user_id,
            )

            await update.message.reply_text(
                "The AI returned an empty response."
            )

            return

        logger.info(
            "AI response received user_id=%s",
            telegram_user_id,
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


    except groq.RateLimitError:

        logger.warning(
            "Groq rate limit reached user_id=%s",
            telegram_user_id,
        )

        await update.message.reply_text(
            "The AI rate limit has been reached. "
            "Please try again shortly."
        )


    except groq.APITimeoutError:

        logger.warning(
            "Groq timeout user_id=%s",
            telegram_user_id,
        )

        await update.message.reply_text(
            "The AI took too long to respond. "
            "Please try again."
        )


    except groq.APIConnectionError:

        logger.exception(
            "Groq connection error user_id=%s",
            telegram_user_id,
        )

        await update.message.reply_text(
            "I couldn't connect to the AI service. "
            "Please try again."
        )


    except Exception:

        logger.exception(
            "Unexpected message-processing "
            "error user_id=%s",
            telegram_user_id,
        )

        await update.message.reply_text(
            "Something went wrong."
        )


# ==================================================
# TELEGRAM STARTUP
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
        "AI model=%s",
        AI_MODEL,
    )

    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

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
