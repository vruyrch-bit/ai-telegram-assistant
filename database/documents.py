import json

import psycopg

from config import (
    DATABASE_URL,
    EMBEDDING_MODEL_NAME,
)


async def save_document(
    telegram_user_id: int,
    filename: str,
    file_type: str,
    chunks,
    embeddings=None,
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

        for index, chunk in enumerate(chunks):
            embedding_json = None
            embedding_model = None

            if (
                embeddings
                and index < len(embeddings)
            ):
                embedding_json = json.dumps(
                    embeddings[index]
                )

                embedding_model = (
                    EMBEDDING_MODEL_NAME
                )

            await connection.execute(
                """
                INSERT INTO document_chunks (
                    document_id,
                    chunk_index,
                    content,
                    embedding,
                    embedding_model
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,
                (
                    document_id,
                    index,
                    chunk,
                    embedding_json,
                    embedding_model,
                ),
            )

    return document_id


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
