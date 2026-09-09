import logging
import sys

from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from config import (
    TELEGRAM_TOKEN,
    AI_MODEL,
    VOICE_MODEL,
    VISION_MODEL,
    EMBEDDING_MODEL_NAME,
)

from database.core import (
    initialize_database,
)

from database.long_term_memory import (
    initialize_long_term_memory,
)

from bot.commands import (
    start,
    clear_memory,
    files_command,
    clear_files,
    clear_image_command,
    tasks_command,
    add_task_command,
    done_task_command,
    delete_task_command,
    remember_command,
    memory_command,
    forget_command,
)

from bot.handlers import (
    handle_message,
    handle_voice,
    handle_photo,
)

from bot.document_handler import (
    handle_document,
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | telegram-bot | "
        "%(name)s | %(message)s"
    ),
    stream=sys.stdout,
    force=True,
)

logging.getLogger(
    "httpx"
).setLevel(
    logging.WARNING
)

logger = logging.getLogger(
    __name__
)


async def post_init(
    application: Application,
):
    await initialize_database()
    await initialize_long_term_memory()

    logger.info(
        "Telegram bot initialization complete"
    )


def build_application():
    application = (
        Application.builder()
        .token(
            TELEGRAM_TOKEN
        )
        .post_init(
            post_init
        )
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

    application.add_handler(
        CommandHandler(
            "clearimage",
            clear_image_command,
        )
    )

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

    application.add_handler(
        MessageHandler(
            filters.VOICE,
            handle_voice,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            handle_photo,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.Document.ALL,
            handle_document,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            handle_message,
        )
    )

    return application


def run_bot():
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

    logger.info(
        "Vision model=%s",
        VISION_MODEL,
    )

    logger.info(
        "Embedding model=%s",
        EMBEDDING_MODEL_NAME,
    )

    application = (
        build_application()
    )

    logger.info(
        "Starting Telegram polling"
    )

    application.run_polling()
