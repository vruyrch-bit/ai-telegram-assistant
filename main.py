import os

import psycopg
import groq

from dotenv import load_dotenv
from groq import AsyncGroq

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)


# ==================================================
# SETTINGS
# ==================================================

AI_MODEL = "openai/gpt-oss-20b"

# All messages remain stored in PostgreSQL.
# Only the latest 30 are sent to the AI each time.
MEMORY_MESSAGE_LIMIT = 30


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")


if not TELEGRAM_TOKEN:
    raise ValueError(
        "TELEGRAM_TOKEN was not found in .env"
    )

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY was not found in .env"
    )

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL was not found in .env"
    )


# ==================================================
# GROQ CLIENT
# ==================================================

# max_retries=1 prevents very long automatic retry delays.
# timeout=20 means a request cannot silently wait for a minute.

client = AsyncGroq(
    api_key=GROQ_API_KEY,
    timeout=20.0,
    max_retries=1
)


# ==================================================
# DATABASE INITIALIZATION
# ==================================================

async def initialize_database():

    print("Connecting to PostgreSQL...")

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

    print("PostgreSQL memory database ready.")


# ==================================================
# SAVE MESSAGE
# ==================================================

async def save_message(
    telegram_user_id: int,
    role: str,
    content: str
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
                content
            )
        )


# ==================================================
# LOAD MEMORY
# ==================================================

async def load_memory(
    telegram_user_id: int
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
                MEMORY_MESSAGE_LIMIT
            )
        )

        rows = await cursor.fetchall()


    # Database returned newest first.
    # AI needs oldest first.
    rows.reverse()

    messages = []


    for role, content in rows:

        # Old Gemini messages were stored as "model".
        # Groq/OpenAI expects "assistant".
        if role == "model":
            role = "assistant"

        # Only allow valid chat roles.
        if role not in [
            "user",
            "assistant",
            "system"
        ]:
            continue

        messages.append(
            {
                "role": role,
                "content": content
            }
        )


    return messages


# ==================================================
# DELETE MEMORY
# ==================================================

async def delete_memory(
    telegram_user_id: int
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
            )
        )


# ==================================================
# /start
# ==================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "Hello! 👋\n\n"
        "I am your AI assistant powered by "
        "OpenAI GPT-OSS 20B.\n\n"
        "I also have persistent PostgreSQL memory.\n\n"
        "Send me anything you'd like to talk about.\n\n"
        "Use /clear to delete our conversation memory."
    )


# ==================================================
# /clear
# ==================================================

async def clear_memory(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    telegram_user_id = (
        update.effective_user.id
    )

    await delete_memory(
        telegram_user_id
    )

    print(
        f"Memory cleared for user "
        f"{telegram_user_id}"
    )

    await update.message.reply_text(
        "Conversation memory cleared. 🧹"
    )


# ==================================================
# SEND LONG TELEGRAM MESSAGE
# ==================================================

async def send_long_message(
    update: Update,
    text: str
):

    # Telegram's maximum message size
    # is slightly above 4000 characters.
    max_length = 4000


    for i in range(
        0,
        len(text),
        max_length
    ):

        part = text[
            i:i + max_length
        ]

        await update.message.reply_text(
            part
        )


# ==================================================
# ASK GROQ / OPENAI GPT-OSS
# ==================================================

async def ask_ai(
    user_message: str,
    history
):

    messages = [
{
    "role": "system",
    "content": (
        "You are an AI assistant running inside a Telegram bot. "
        "You are powered by OpenAI's GPT-OSS 20B model through Groq. "
        "You are not ChatGPT and should not claim to be ChatGPT. "
        "If asked who or what you are, explain that you are a custom "
        "AI Telegram assistant built using GPT-OSS 20B. "
        "Give clear, useful answers. "
        "Keep answers reasonably concise unless the user asks for detail."
    )
}
    ]

    # Add previous conversation
    messages.extend(history)

    # Add newest user message
    messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )


    print(
        f"Sending request to {AI_MODEL}..."
    )


    response = await client.chat.completions.create(
        model=AI_MODEL,
        messages=messages,

        # GPT-OSS supports low/medium/high reasoning.
        # Low keeps ordinary Telegram answers faster.
        reasoning_effort="low",

        # Prevent extremely huge responses.
        max_completion_tokens=1500
    )


    answer = (
        response
        .choices[0]
        .message
        .content
    )


    return answer


# ==================================================
# HANDLE USER MESSAGE
# ==================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    telegram_user_id = (
        update.effective_user.id
    )

    user_message = (
        update.message.text
    )


    print(
        f"Message from user "
        f"{telegram_user_id}: "
        f"{user_message}"
    )


    try:

        # ------------------------------------------
        # Telegram typing animation
        # ------------------------------------------

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING
        )


        # ------------------------------------------
        # Load PostgreSQL memory
        # ------------------------------------------

        history = await load_memory(
            telegram_user_id
        )


        print(
            f"Loaded {len(history)} "
            f"messages from PostgreSQL."
        )


        # ------------------------------------------
        # Ask AI
        # ------------------------------------------

        answer = await ask_ai(
            user_message,
            history
        )


        if not answer:

            await update.message.reply_text(
                "The AI returned an empty response."
            )

            return


        print(
            "AI response received."
        )


        # ------------------------------------------
        # Save user message
        # ------------------------------------------

        await save_message(
            telegram_user_id,
            "user",
            user_message
        )


        # ------------------------------------------
        # Save AI response
        # ------------------------------------------

        await save_message(
            telegram_user_id,
            "assistant",
            answer
        )


        print(
            "Conversation saved to PostgreSQL."
        )


        # ------------------------------------------
        # Send answer
        # ------------------------------------------

        await send_long_message(
            update,
            answer
        )


    # ==================================================
    # GROQ RATE LIMIT
    # ==================================================

    except groq.RateLimitError as error:

        print(
            f"GROQ RATE LIMIT: {error}"
        )

        await update.message.reply_text(
            "The AI rate limit has been reached. "
            "Please try again shortly."
        )


    # ==================================================
    # GROQ TIMEOUT
    # ==================================================

    except groq.APITimeoutError as error:

        print(
            f"GROQ TIMEOUT: {error}"
        )

        await update.message.reply_text(
            "The AI took too long to respond. "
            "Please try again."
        )


    # ==================================================
    # CONNECTION ERROR
    # ==================================================

    except groq.APIConnectionError as error:

        print(
            f"GROQ CONNECTION ERROR: {error}"
        )

        await update.message.reply_text(
            "I couldn't connect to the AI service. "
            "Please try again."
        )


    # ==================================================
    # OTHER ERRORS
    # ==================================================

    except Exception as error:

        print(
            f"FINAL ERROR: {error}"
        )

        await update.message.reply_text(
            "Something went wrong."
        )


# ==================================================
# TELEGRAM STARTUP
# ==================================================

async def post_init(
    application: Application
):

    await initialize_database()


# ==================================================
# RUN BOT
# ==================================================

def main():

    print(
        "Starting AI Telegram Bot..."
    )

    print(
        f"AI model: {AI_MODEL}"
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
            start
        )
    )


    application.add_handler(
        CommandHandler(
            "clear",
            clear_memory
        )
    )


    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            handle_message
        )
    )


    print(
        "Starting Telegram connection..."
    )


    application.run_polling()


# ==================================================
# START PROGRAM
# ==================================================

if __name__ == "__main__":
    main()
