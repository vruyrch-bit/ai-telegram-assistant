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
            i:
            i + max_length
        ]

        await update.message.reply_text(
            part
        )
