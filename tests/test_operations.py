"""Operational tests restricted to the disposable PostgreSQL fixture."""
import os
from pathlib import Path
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def empty_database(isolated_db):
    import config
    name = 'bot_test_' + uuid.uuid4().hex
    async with await psycopg.AsyncConnection.connect(config.DATABASE_URL, autocommit=True) as connection:
        await connection.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
        try:
            yield make_conninfo(config.DATABASE_URL, dbname=name)
        finally:
            await connection.execute(sql.SQL('DROP DATABASE {}').format(sql.Identifier(name)))


@pytest.mark.asyncio
async def test_empty_database_migrations_are_repeatable(empty_database, monkeypatch):
    from database import core, long_term_memory, upgrades
    for module in (core, long_term_memory, upgrades):
        monkeypatch.setattr(module, 'DATABASE_URL', empty_database)
    async def migrate():
        await core.initialize_database()
        await long_term_memory.initialize_long_term_memory()
        await upgrades.initialize_upgrades()
    await migrate()
    async with await psycopg.AsyncConnection.connect(empty_database) as connection:
        await connection.execute("INSERT INTO knowledge_notes (telegram_user_id, title, content) VALUES (101, 'fixture', 'preserved')")
    await migrate()
    async with await psycopg.AsyncConnection.connect(empty_database) as connection:
        cursor = await connection.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        tables = {row[0] for row in await cursor.fetchall()}
        assert tables == {'messages', 'documents', 'document_chunks', 'tasks', 'latest_images',
                          'long_term_memories', 'user_preferences', 'reminders', 'knowledge_notes',
                          'tool_events', 'image_history'}
        cursor = await connection.execute("SELECT indexname FROM pg_indexes WHERE schemaname = 'public'")
        indexes = {row[0] for row in await cursor.fetchall()}
        assert {'reminders_due_idx', 'reminders_user_idx', 'notes_user_idx', 'tool_events_time_idx',
                'image_history_user_idx', 'idx_documents_user', 'idx_tasks_user'} <= indexes
        cursor = await connection.execute('SELECT content FROM knowledge_notes')
        assert await cursor.fetchall() == [('preserved',)]


@pytest.mark.asyncio
async def test_backup_restore_private_permissions_and_overwrite_refusal(isolated_db, empty_database, tmp_path):
    import config
    await isolated_db("INSERT INTO knowledge_notes (telegram_user_id, title, content) VALUES (101, 'fixture', 'restore marker')")
    await isolated_db("INSERT INTO tasks (telegram_user_id, title, priority, project) VALUES (101, 'Guitar', 5, 'guitar')")
    dump = tmp_path / 'fixture.dump'
    command = [sys.executable, 'scripts/backup_database.py', str(dump)]
    env = {**os.environ, 'DATABASE_URL': config.DATABASE_URL}
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, 'Disposable database backup failed'
    assert dump.stat().st_mode & 0o777 == 0o600
    before = dump.read_bytes()
    assert subprocess.run(command, env=env, capture_output=True).returncode != 0
    assert dump.read_bytes() == before
    parts = conninfo_to_dict(empty_database)
    restore_env = {key: value for key, value in os.environ.items() if not key.startswith('PG')}
    restore_env.update({{'host': 'PGHOST', 'port': 'PGPORT', 'user': 'PGUSER', 'password': 'PGPASSWORD',
                        'dbname': 'PGDATABASE'}[key]: value for key, value in parts.items()})
    result = subprocess.run(['pg_restore', '--exit-on-error', '--no-owner', '--no-acl',
                             '--dbname', parts['dbname'], str(dump)], env=restore_env, capture_output=True)
    assert result.returncode == 0, 'Disposable database restore failed'
    async with await psycopg.AsyncConnection.connect(empty_database) as connection:
        cursor = await connection.execute('SELECT content FROM knowledge_notes')
        assert await cursor.fetchall() == [('restore marker',)]
        cursor = await connection.execute('SELECT priority, project FROM tasks')
        assert await cursor.fetchall() == [(5, 'guitar')]
        cursor = await connection.execute("INSERT INTO knowledge_notes (telegram_user_id, title, content) VALUES (101, 'next', 'sequence') RETURNING id")
        assert (await cursor.fetchone())[0] == 2


@pytest.mark.asyncio
async def test_worker_retry_exhaustion(isolated_db):
    from services.jobs import deliver_one
    await isolated_db("INSERT INTO reminders (telegram_user_id, content, due_at, timezone) VALUES (101, 'Fixture', %s, 'UTC')",
                      (datetime.now(timezone.utc) - timedelta(minutes=1),))
    bot = AsyncMock()
    bot.send_message.side_effect = RuntimeError('private provider error must not be stored')
    for attempt in range(1, 6):
        assert await deliver_one(bot)
        row = (await isolated_db('SELECT * FROM reminders'))[0]
        assert row['attempts'] == attempt
        assert row['status'] == ('failed' if attempt == 5 else 'pending')
        assert row['last_error'] == 'RuntimeError'
        await isolated_db('UPDATE reminders SET retry_at = NULL')
    assert not await deliver_one(bot)
    assert bot.send_message.await_count == 5


@pytest.mark.asyncio
async def test_weekly_reminder_preserves_zone_and_resets_retry(isolated_db):
    from services.jobs import deliver_one
    from zoneinfo import ZoneInfo
    due = datetime.now(ZoneInfo('Asia/Yerevan')) - timedelta(minutes=1)
    await isolated_db("""INSERT INTO reminders (telegram_user_id, content, due_at, timezone, recurrence, attempts, last_error)
        VALUES (101, 'Fixture', %s, 'Asia/Yerevan', 'weekly', 2, 'RuntimeError')""", (due,))
    assert await deliver_one(AsyncMock())
    row = (await isolated_db('SELECT * FROM reminders'))[0]
    assert row['due_at'] == due + timedelta(days=7)
    assert row['status'] == 'pending' and row['attempts'] == 0 and row['last_error'] is None
