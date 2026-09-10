"""Additive, repeatable migrations for personal assistant features."""
import psycopg
from config import DATABASE_URL


async def initialize_upgrades():
    statements = [
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS collection TEXT NOT NULL DEFAULT ''",
        """CREATE TABLE IF NOT EXISTS user_preferences (
            telegram_user_id BIGINT PRIMARY KEY, timezone TEXT NOT NULL)""",
        "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS priority SMALLINT NOT NULL DEFAULT 3 CHECK (priority BETWEEN 1 AND 5)",
        "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS project TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS recurrence TEXT NOT NULL DEFAULT 'none' CHECK (recurrence IN ('none', 'daily', 'weekly'))",
        "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS notes TEXT NOT NULL DEFAULT ''",
        """CREATE TABLE IF NOT EXISTS reminders (
            id BIGSERIAL PRIMARY KEY, telegram_user_id BIGINT NOT NULL,
            content TEXT NOT NULL, due_at TIMESTAMPTZ NOT NULL,
            timezone TEXT NOT NULL, recurrence TEXT NOT NULL DEFAULT 'none'
                CHECK (recurrence IN ('none', 'daily', 'weekly')),
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'sent', 'cancelled', 'failed')),
            attempts INTEGER NOT NULL DEFAULT 0, retry_at TIMESTAMPTZ,
            last_error TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())""",
        "CREATE INDEX IF NOT EXISTS reminders_due_idx ON reminders (due_at) WHERE status = 'pending'",
        "CREATE INDEX IF NOT EXISTS reminders_user_idx ON reminders (telegram_user_id, id)",
        """CREATE TABLE IF NOT EXISTS knowledge_notes (
            id BIGSERIAL PRIMARY KEY, telegram_user_id BIGINT NOT NULL,
            title TEXT NOT NULL, content TEXT NOT NULL, project TEXT NOT NULL DEFAULT '',
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())""",
        "CREATE INDEX IF NOT EXISTS notes_user_idx ON knowledge_notes (telegram_user_id, id)",
        """CREATE TABLE IF NOT EXISTS tool_events (
            id BIGSERIAL PRIMARY KEY, telegram_user_id BIGINT NOT NULL,
            tool TEXT NOT NULL, success BOOLEAN NOT NULL, duration_ms INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())""",
        "CREATE INDEX IF NOT EXISTS tool_events_time_idx ON tool_events (created_at)",
        """CREATE TABLE IF NOT EXISTS image_history (
            id BIGSERIAL PRIMARY KEY, telegram_user_id BIGINT NOT NULL,
            filename TEXT NOT NULL, mime_type TEXT NOT NULL, image_data BYTEA NOT NULL,
            ocr_text TEXT, vision_summary TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())""",
        "CREATE INDEX IF NOT EXISTS image_history_user_idx ON image_history (telegram_user_id, id DESC)",
    ]
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as connection:
        for statement in statements:
            await connection.execute(statement)
