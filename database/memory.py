import psycopg

from config import (
    DATABASE_URL,
    MEMORY_MESSAGE_LIMIT,
)


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
