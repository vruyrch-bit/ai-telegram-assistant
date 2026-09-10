import re

from telegram import Update


def clean_telegram_text(
    text: str,
):
    text = text.replace(
        "**",
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


async def send_long_message(
    update: Update,
    text: str,
):
    text = clean_telegram_text(
        text
    )

    max_length = 4000

    # Count UTF-16 units so astral characters (including emoji) fit too.
    start = 0
    units = 0
    for index, character in enumerate(text):
        width = 2 if ord(character) > 0xFFFF else 1
        if units + width > max_length:
            await update.message.reply_text(text[start:index])
            start = index
            units = 0
        units += width
    if start < len(text):
        await update.message.reply_text(text[start:])
