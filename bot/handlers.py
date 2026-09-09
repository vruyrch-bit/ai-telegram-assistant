import logging

import groq

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from config import (
    MAX_VOICE_SIZE,
)

from database.memory import (
    save_message,
    load_memory,
)

from database.images import (
    load_latest_image,
)

from rag.retrieval import (
    get_document_context,
)

from services.ai import (
    ask_ai,
)

from services.memory import (
    build_memory_context,
)

from services.vision import (
    should_use_latest_image,
    analyze_image_with_vision,
    process_image_upload,
)

from services.voice import (
    transcribe_voice,
)

from utils.telegram_text import (
    clean_telegram_text,
    send_long_message,
)


logger = logging.getLogger(__name__)


# ==================================================
# USER MESSAGE PROCESSING
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

    # If the user explicitly refers to the latest image,
    # send the latest stored image back through the vision model.
    if should_use_latest_image(
        user_message
    ):
        latest_image = (
            await load_latest_image(
                telegram_user_id
            )
        )

        if latest_image:
            (
                filename,
                mime_type,
                image_data,
                ocr_text,
                vision_summary,
            ) = latest_image

            image_answer = (
                await analyze_image_with_vision(
                    bytes(image_data),
                    user_message,
                    ocr_text or "",
                )
            )

            image_answer = clean_telegram_text(
                image_answer
            )

            await save_message(
                telegram_user_id,
                "user",
                user_message,
            )

            await save_message(
                telegram_user_id,
                "assistant",
                image_answer,
            )

            await send_long_message(
                update,
                image_answer,
            )

            return

    history = await load_memory(
        telegram_user_id
    )

    memory_context = (
        await build_memory_context(
            telegram_user_id,
            user_message,
        )
    )

    document_context = (
        await get_document_context(
            telegram_user_id,
            user_message,
        )
    )

    answer = await ask_ai(
        telegram_user_id,
        user_message,
        history,
        document_context,
        memory_context,
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

    await send_long_message(
        update,
        answer,
    )


# ==================================================
# TEXT MESSAGES
# ==================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    telegram_user_id = (
        update.effective_user.id
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

    except Exception:
        logger.exception(
            "Text processing error "
            "user_id=%s",
            telegram_user_id,
        )

        await update.message.reply_text(
            "Something went wrong."
        )


# ==================================================
# VOICE MESSAGES
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
            await telegram_file
            .download_as_bytearray()
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
            "Voice processing error "
            "user_id=%s",
            telegram_user_id,
        )

        await update.message.reply_text(
            "I couldn't process that voice message."
        )


# ==================================================
# PHOTO MESSAGES
# ==================================================

async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    photo = (
        update.message.photo[-1]
    )

    await process_image_upload(
        update=update,
        context=context,
        file_id=photo.file_id,
        filename="telegram_photo.jpg",
        file_size=photo.file_size,
        caption=update.message.caption,
    )
