import psycopg

from config import DATABASE_URL


async def save_latest_image(
    telegram_user_id: int,
    filename: str,
    mime_type: str,
    image_data: bytes,
    ocr_text: str,
    vision_summary: str,
):
    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:
        await connection.execute(
            """
            INSERT INTO latest_images (
                telegram_user_id,
                filename,
                mime_type,
                image_data,
                ocr_text,
                vision_summary
            )
            VALUES (%s, %s, %s, %s, %s, %s)

            ON CONFLICT (telegram_user_id)
            DO UPDATE SET
                filename = EXCLUDED.filename,
                mime_type = EXCLUDED.mime_type,
                image_data = EXCLUDED.image_data,
                ocr_text = EXCLUDED.ocr_text,
                vision_summary = EXCLUDED.vision_summary,
                created_at = CURRENT_TIMESTAMP
            """,
            (
                telegram_user_id,
                filename,
                mime_type,
                image_data,
                ocr_text,
                vision_summary,
            ),
        )


async def load_latest_image(
    telegram_user_id: int,
):
    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:
        cursor = await connection.execute(
            """
            SELECT
                filename,
                mime_type,
                image_data,
                ocr_text,
                vision_summary
            FROM latest_images
            WHERE telegram_user_id = %s
            """,
            (
                telegram_user_id,
            ),
        )

        return await cursor.fetchone()


async def delete_latest_image(
    telegram_user_id: int,
):
    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:
        await connection.execute(
            """
            DELETE FROM latest_images
            WHERE telegram_user_id = %s
            """,
            (
                telegram_user_id,
            ),
        )
