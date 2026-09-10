import asyncio
import base64
import io
import logging

from groq import AsyncGroq
from PIL import Image, ImageOps
from telegram import Update
from telegram.ext import ContextTypes

from config import (
    GROQ_API_KEY,
    VISION_MODEL,
    MAX_IMAGE_SIZE,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_OCR_CONTEXT,
)

from database.images import save_latest_image
from database.memory import save_message
from services.ocr import ocr_image


logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


IMAGE_REFERENCE_PHRASES = (
    "this image",
    "the image",
    "this photo",
    "the photo",
    "this picture",
    "the picture",
    "this screenshot",
    "the screenshot",
    "in the image",
    "in the photo",
    "in the picture",
    "in the screenshot",
)


from config import AI_PROVIDER
from services.local_ai import LocalClient

vision_client = LocalClient() if AI_PROVIDER == "local" else AsyncGroq(
    api_key=GROQ_API_KEY, timeout=30.0, max_retries=1,
)


def should_use_latest_image(
    user_message: str,
):
    lower_message = user_message.lower()

    return any(
        phrase in lower_message
        for phrase in IMAGE_REFERENCE_PHRASES
    )


def normalize_image_for_vision(
    image_bytes: bytes,
):
    image = Image.open(
        io.BytesIO(image_bytes)
    )

    image = ImageOps.exif_transpose(
        image
    )

    image.thumbnail(
        (
            MAX_IMAGE_DIMENSION,
            MAX_IMAGE_DIMENSION,
        )
    )

    if image.mode in (
        "RGBA",
        "LA",
    ):
        background = Image.new(
            "RGB",
            image.size,
            "white",
        )

        alpha = image.getchannel(
            "A"
        )

        background.paste(
            image,
            mask=alpha,
        )

        image = background

    elif image.mode != "RGB":
        image = image.convert(
            "RGB"
        )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=88,
        optimize=True,
    )

    return (
        buffer.getvalue(),
        image,
    )


async def analyze_image_with_vision(
    image_bytes: bytes,
    user_prompt: str,
    ocr_text: str = "",
):
    base64_image = base64.b64encode(
        image_bytes
    ).decode(
        "utf-8"
    )

    prompt = (
        "Analyze the attached image only according to the "
        "user's specific request. Do not automatically give "
        "a full description unless the user asks for one. "
        "Answer using only what is visibly supported by the "
        "image. You may describe objects, scenes, diagrams, "
        "screenshots, documents, products, colors, layout, "
        "and visible text when relevant to the request. "
        "Do not guess the identity of real people or identify "
        "specific TV/movie characters from the image. "
        "If something is uncertain, say that it is uncertain. "
        "Be concise by default. Give a longer answer only if "
        "the user asks for detail or asks you to read or "
        "transcribe a large amount of text. Use plain text "
        "suitable for Telegram. Do not use Markdown tables or "
        "Markdown formatting symbols.\n\n"
        f"User request:\n{user_prompt}"
    )

    if ocr_text:
        prompt += (
            "\n\nLocal OCR detected the following text. "
            "Treat it only as a hint because OCR can contain "
            "mistakes. Prefer the actual image when they "
            "conflict:\n"
            + ocr_text[
                :MAX_IMAGE_OCR_CONTEXT
            ]
        )

    response = (
        await vision_client
        .chat
        .completions
        .create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/jpeg;base64,"
                                    + base64_image
                                ),
                            },
                        },
                    ],
                }
            ],
            temperature=0.2,
            max_completion_tokens=1200,
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
        or ""
    ).strip()


async def process_image_upload(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    file_id: str,
    filename: str,
    file_size,
    caption: str | None,
):
    telegram_user_id = (
        update.effective_user.id
    )

    if (
        file_size
        and file_size > MAX_IMAGE_SIZE
    ):
        await update.message.reply_text(
            "That image is too large."
        )
        return

    telegram_file = (
        await context.bot.get_file(
            file_id
        )
    )

    image_data = (
        await telegram_file
        .download_as_bytearray()
    )

    if len(image_data) > MAX_IMAGE_SIZE:
        await update.message.reply_text("That image is too large.")
        return

    original_bytes = bytes(
        image_data
    )

    try:
        (
            normalized_bytes,
            normalized_image,
        ) = await asyncio.to_thread(
            normalize_image_for_vision,
            original_bytes,
        )

    except Exception:
        logger.exception(
            "Image normalization failed "
            "user_id=%s",
            telegram_user_id,
        )

        await update.message.reply_text(
            "I couldn't open that image."
        )
        return

    try:
        ocr_text = (
            await asyncio.to_thread(
                ocr_image,
                normalized_image,
            )
        )

    except Exception:
        logger.exception(
            "Image OCR failed "
            "user_id=%s",
            telegram_user_id,
        )

        ocr_text = ""

    await save_latest_image(
        telegram_user_id,
        filename,
        "image/jpeg",
        normalized_bytes,
        ocr_text,
        "",
    )

    await save_message(
        telegram_user_id,
        "user",
        "[Image uploaded]",
    )

    await update.message.reply_text(
        "🖼️ Image received. "
        "Tell me what you'd like me to do with it."
    )
