import hashlib
import json

import psycopg

from config import (
    DATABASE_URL,
    EMBEDDING_MODEL_NAME,
)


# ==================================================
# HELPERS
# ==================================================

def memory_content_hash(
    content: str,
):
    normalized = (
        " ".join(
            content
            .strip()
            .lower()
            .split()
        )
    )

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


# ==================================================
# TABLE INITIALIZATION
# ==================================================

async def initialize_long_term_memory():
    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        async with connection.cursor() as cursor:

            await cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id BIGSERIAL PRIMARY KEY,

                    telegram_user_id BIGINT NOT NULL,

                    content TEXT NOT NULL,

                    memory_type TEXT NOT NULL
                        DEFAULT 'fact',

                    importance SMALLINT NOT NULL
                        DEFAULT 3
                        CHECK (
                            importance >= 1
                            AND importance <= 5
                        ),

                    content_hash TEXT NOT NULL,

                    embedding TEXT,

                    embedding_model TEXT,

                    source TEXT NOT NULL
                        DEFAULT 'user',

                    is_active BOOLEAN NOT NULL
                        DEFAULT TRUE,

                    created_at TIMESTAMPTZ NOT NULL
                        DEFAULT NOW(),

                    updated_at TIMESTAMPTZ NOT NULL
                        DEFAULT NOW(),

                    last_accessed_at TIMESTAMPTZ,

                    UNIQUE (
                        telegram_user_id,
                        content_hash
                    )
                )
                """
            )

            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS
                long_term_memories_user_active_idx
                ON long_term_memories (
                    telegram_user_id,
                    is_active,
                    id DESC
                )
                """
            )

            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS
                long_term_memories_type_idx
                ON long_term_memories (
                    telegram_user_id,
                    memory_type
                )
                """
            )

        await connection.commit()


# ==================================================
# SAVE MEMORY
# ==================================================

async def save_long_term_memory(
    telegram_user_id: int,
    content: str,
    memory_type: str = "fact",
    importance: int = 3,
    embedding=None,
    source: str = "user",
):
    content = content.strip()

    if not content:
        raise ValueError(
            "Memory content cannot be empty."
        )

    importance = max(
        1,
        min(
            int(importance),
            5,
        ),
    )

    content_hash = memory_content_hash(
        content
    )

    embedding_json = None

    if embedding is not None:
        embedding_json = json.dumps(
            [
                float(value)
                for value in embedding
            ]
        )

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        async with connection.cursor() as cursor:

            await cursor.execute(
                """
                INSERT INTO long_term_memories (
                    telegram_user_id,
                    content,
                    memory_type,
                    importance,
                    content_hash,
                    embedding,
                    embedding_model,
                    source,
                    is_active
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    TRUE
                )

                ON CONFLICT (
                    telegram_user_id,
                    content_hash
                )
                DO UPDATE SET
                    content = EXCLUDED.content,
                    memory_type =
                        EXCLUDED.memory_type,
                    importance =
                        EXCLUDED.importance,
                    embedding =
                        COALESCE(
                            EXCLUDED.embedding,
                            long_term_memories.embedding
                        ),
                    embedding_model =
                        COALESCE(
                            EXCLUDED.embedding_model,
                            long_term_memories.embedding_model
                        ),
                    source =
                        EXCLUDED.source,
                    is_active = TRUE,
                    updated_at = NOW()

                RETURNING id
                """,
                (
                    telegram_user_id,
                    content,
                    memory_type,
                    importance,
                    content_hash,
                    embedding_json,
                    (
                        EMBEDDING_MODEL_NAME
                        if embedding is not None
                        else None
                    ),
                    source,
                ),
            )

            row = await cursor.fetchone()

        await connection.commit()

    return row[0]


# ==================================================
# LIST MEMORIES
# ==================================================

async def list_long_term_memories(
    telegram_user_id: int,
    limit: int = 50,
):
    limit = max(
        1,
        min(
            int(limit),
            200,
        ),
    )

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        async with connection.cursor() as cursor:

            await cursor.execute(
                """
                SELECT
                    id,
                    content,
                    memory_type,
                    importance,
                    source,
                    created_at,
                    updated_at
                FROM long_term_memories

                WHERE telegram_user_id = %s
                AND is_active = TRUE

                ORDER BY
                    importance DESC,
                    updated_at DESC

                LIMIT %s
                """,
                (
                    telegram_user_id,
                    limit,
                ),
            )

            rows = await cursor.fetchall()

    return rows


# ==================================================
# GET MEMORY
# ==================================================

async def get_long_term_memory(
    telegram_user_id: int,
    memory_id: int,
):
    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        async with connection.cursor() as cursor:

            await cursor.execute(
                """
                SELECT
                    id,
                    content,
                    memory_type,
                    importance,
                    embedding,
                    embedding_model,
                    source,
                    created_at,
                    updated_at,
                    last_accessed_at
                FROM long_term_memories

                WHERE telegram_user_id = %s
                AND id = %s
                AND is_active = TRUE
                """,
                (
                    telegram_user_id,
                    memory_id,
                ),
            )

            row = await cursor.fetchone()

    return row


# ==================================================
# FORGET ONE MEMORY
# ==================================================

async def forget_long_term_memory(
    telegram_user_id: int,
    memory_id: int,
):
    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        async with connection.cursor() as cursor:

            await cursor.execute(
                """
                UPDATE long_term_memories

                SET
                    is_active = FALSE,
                    updated_at = NOW()

                WHERE telegram_user_id = %s
                AND id = %s
                AND is_active = TRUE

                RETURNING content
                """,
                (
                    telegram_user_id,
                    memory_id,
                ),
            )

            row = await cursor.fetchone()

        await connection.commit()

    if not row:
        return None

    return row[0]


# ==================================================
# CLEAR ALL LONG-TERM MEMORY
# ==================================================

async def clear_long_term_memories(
    telegram_user_id: int,
):
    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        async with connection.cursor() as cursor:

            await cursor.execute(
                """
                UPDATE long_term_memories

                SET
                    is_active = FALSE,
                    updated_at = NOW()

                WHERE telegram_user_id = %s
                AND is_active = TRUE
                """,
                (
                    telegram_user_id,
                ),
            )

            affected = (
                cursor.rowcount
            )

        await connection.commit()

    return affected


# ==================================================
# LIST MEMORIES WITH EMBEDDINGS
# ==================================================

async def list_long_term_memories_with_embeddings(
    telegram_user_id: int,
    limit: int = 100,
):
    limit = max(
        1,
        min(
            int(limit),
            200,
        ),
    )

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        async with connection.cursor() as cursor:

            await cursor.execute(
                """
                SELECT
                    id,
                    content,
                    memory_type,
                    importance,
                    embedding,
                    embedding_model,
                    source,
                    updated_at
                FROM long_term_memories

                WHERE telegram_user_id = %s
                AND is_active = TRUE

                ORDER BY
                    importance DESC,
                    updated_at DESC

                LIMIT %s
                """,
                (
                    telegram_user_id,
                    limit,
                ),
            )

            rows = await cursor.fetchall()

    return rows


# ==================================================
# REPLACE MEMORY
# ==================================================

async def replace_long_term_memory(
    telegram_user_id: int,
    memory_id: int,
    content: str,
    memory_type: str,
    importance: int,
    embedding=None,
    source: str = "automatic",
):
    content = content.strip()

    if not content:
        raise ValueError(
            "Memory content cannot be empty."
        )

    importance = max(
        1,
        min(
            int(importance),
            5,
        ),
    )

    content_hash = memory_content_hash(
        content
    )

    embedding_json = None

    if embedding is not None:
        embedding_json = json.dumps(
            [
                float(value)
                for value in embedding
            ]
        )

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        async with connection.cursor() as cursor:

            await cursor.execute(
                """
                UPDATE long_term_memories

                SET
                    content = %s,
                    memory_type = %s,
                    importance = %s,
                    content_hash = %s,
                    embedding = %s,
                    embedding_model = %s,
                    source = %s,
                    updated_at = NOW()

                WHERE telegram_user_id = %s
                AND id = %s
                AND is_active = TRUE

                RETURNING id
                """,
                (
                    content,
                    memory_type,
                    importance,
                    content_hash,
                    embedding_json,
                    (
                        EMBEDDING_MODEL_NAME
                        if embedding is not None
                        else None
                    ),
                    source,
                    telegram_user_id,
                    memory_id,
                ),
            )

            row = await cursor.fetchone()

        await connection.commit()

    if not row:
        return None

    return row[0]
