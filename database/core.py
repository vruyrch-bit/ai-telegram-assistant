import logging

import psycopg

from config import DATABASE_URL


logger = logging.getLogger(__name__)


async def initialize_database():
    logger.info("Connecting to PostgreSQL")

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
            CREATE INDEX IF NOT EXISTS idx_messages_user
            ON messages (telegram_user_id, id)
            """
        )

        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id BIGSERIAL PRIMARY KEY,
                telegram_user_id BIGINT NOT NULL,
                filename TEXT NOT NULL,
                file_type VARCHAR(20) NOT NULL,
                created_at TIMESTAMPTZ
                    DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id BIGSERIAL PRIMARY KEY,
                document_id BIGINT NOT NULL
                    REFERENCES documents(id)
                    ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                embedding_model TEXT
            )
            """
        )

        await connection.execute(
            """
            ALTER TABLE document_chunks
            ADD COLUMN IF NOT EXISTS embedding TEXT
            """
        )

        await connection.execute(
            """
            ALTER TABLE document_chunks
            ADD COLUMN IF NOT EXISTS embedding_model TEXT
            """
        )

        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_user
            ON documents (telegram_user_id, id)
            """
        )

        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_document_chunks_document
            ON document_chunks (document_id, chunk_index)
            """
        )

        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id BIGSERIAL PRIMARY KEY,
                telegram_user_id BIGINT NOT NULL,
                title TEXT NOT NULL,
                status VARCHAR(20)
                    NOT NULL
                    DEFAULT 'open',
                due_date TEXT,
                created_at TIMESTAMPTZ
                    DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMPTZ
            )
            """
        )

        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_user
            ON tasks (telegram_user_id, id)
            """
        )

        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS latest_images (
                telegram_user_id BIGINT PRIMARY KEY,
                filename TEXT NOT NULL,
                mime_type VARCHAR(50) NOT NULL,
                image_data BYTEA NOT NULL,
                ocr_text TEXT,
                vision_summary TEXT,
                created_at TIMESTAMPTZ
                    DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    logger.info("PostgreSQL database ready")
