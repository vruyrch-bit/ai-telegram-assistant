"""Opt-in integration tests; use ONLY a disposable database.

Set BOT_TEST_DATABASE_URL to a local isolated database created for this suite.
No production URL is read or used by this test module.
"""
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

URL = os.getenv('BOT_TEST_DATABASE_URL')
pytestmark = pytest.mark.skipif(not URL, reason='Requires isolated BOT_TEST_DATABASE_URL')


@pytest_asyncio.fixture
async def database(isolated_db):
    from database import personal
    return personal


@pytest.mark.asyncio
async def test_task_notes_and_user_isolation(database):
    from database.tasks import create_task, complete_task
    from tools.personal_tools import execute_personal_tool
    task_id = await create_task(101, 'Study')
    result = json.loads(await execute_personal_tool(101, 'edit_task', {'task_id': task_id, 'priority': 5, 'project': 'School', 'notes': 'chapter 2', 'due_date': '2030-01-01', 'recurrence': 'weekly'}))
    assert result['success']
    result = json.loads(await execute_personal_tool(202, 'edit_task', {'task_id': task_id, 'title': 'Stolen'}))
    assert not result['success']
    assert await complete_task(101, task_id) == 'Study'
    assert await complete_task(101, task_id) is None
    rows = await database.fetch('SELECT * FROM tasks WHERE telegram_user_id = 101 AND status = %s', ('open',))
    assert len(rows) == 1 and rows[0]['due_date'] == '2030-01-08'
    result = json.loads(await execute_personal_tool(101, 'save_note', {'title': 'Goal', 'content': 'Learn Python'}))
    note_id = result['results'][0]['id']
    denied = json.loads(await execute_personal_tool(202, 'delete_note', {'note_id': note_id}))
    assert not denied['success']
    rows = json.loads(await execute_personal_tool(101, 'search_notes', {'query': 'Python'}))['results']
    assert rows[0]['title'] == 'Goal'


async def due_job(database, recurrence='none'):
    rows = await database.fetch('''INSERT INTO reminders (telegram_user_id, content, due_at, timezone, recurrence)
        VALUES (101, 'Test', %s, 'UTC', %s) RETURNING id''', (datetime.now(timezone.utc) - timedelta(minutes=1), recurrence))
    return rows[0]['id']


@pytest.mark.asyncio
async def test_concurrent_workers_deliver_once(database):
    from services.jobs import deliver_one
    await due_job(database)
    bot = AsyncMock()
    await asyncio.gather(deliver_one(bot), deliver_one(bot))
    bot.send_message.assert_awaited_once()
    rows = await database.fetch('SELECT status FROM reminders WHERE telegram_user_id = 101')
    assert rows[0]['status'] == 'sent'


@pytest.mark.asyncio
async def test_retry_and_recurrence(database):
    from services.jobs import deliver_one
    job_id = await due_job(database, 'daily')
    bot = AsyncMock()
    bot.send_message.side_effect = RuntimeError('offline')
    assert await deliver_one(bot)
    rows = await database.fetch('SELECT * FROM reminders WHERE id = %s', (job_id,))
    assert rows[0]['status'] == 'pending' and rows[0]['attempts'] == 1
    assert rows[0]['retry_at'] > datetime.now(timezone.utc)
    await database.fetch('UPDATE reminders SET retry_at = NULL WHERE id = %s', (job_id,))
    bot.send_message.side_effect = None
    assert await deliver_one(bot)
    rows = await database.fetch('SELECT * FROM reminders WHERE id = %s', (job_id,))
    assert rows[0]['due_at'] > datetime.now(timezone.utc)
    assert rows[0]['attempts'] == 0


@pytest.mark.asyncio
async def test_cancelled_reminder_never_delivered(database):
    from services.jobs import deliver_one
    from tools.personal_tools import execute_personal_tool
    job_id = await due_job(database)
    denied = json.loads(await execute_personal_tool(202, 'cancel_reminder', {'reminder_id': job_id}))
    assert not denied['success']
    cancelled = json.loads(await execute_personal_tool(101, 'cancel_reminder', {'reminder_id': job_id}))
    assert cancelled['success']
    bot = AsyncMock()
    assert not await deliver_one(bot)
    bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_image_history_retention(database):
    from database.images import save_latest_image, load_latest_image
    for i in range(12):
        await save_latest_image(101, f'{i}.png', 'image/png', b'test-image', '', '')
    rows = await database.fetch('SELECT filename FROM image_history WHERE telegram_user_id = 101 ORDER BY id')
    assert len(rows) == 10 and rows[0]['filename'] == '2.png'
    assert (await load_latest_image(101))[0] == '11.png'
    assert await load_latest_image(202) is None


@pytest.mark.asyncio
async def test_document_collection_search_is_scoped(database, monkeypatch):
    from tools.personal_tools import execute_personal_tool
    from rag import retrieval
    monkeypatch.setattr(retrieval, 'DATABASE_URL', URL)
    rows = await database.fetch("INSERT INTO documents (telegram_user_id, filename, file_type) VALUES (101, 'scoped-test.txt', 'txt') RETURNING id")
    doc_id = rows[0]['id']
    try:
        await database.fetch('INSERT INTO document_chunks (document_id, chunk_index, content) VALUES (%s, 0, %s)', (doc_id, 'My test document.'))
        result = json.loads(await execute_personal_tool(101, 'set_document_collection', {'document_id': doc_id, 'collection': 'school'}))
        assert result['success']
        result = json.loads(await execute_personal_tool(101, 'search_documents', {'query': 'summarize', 'document_id': doc_id, 'collection': 'school'}))
        assert 'My test document.' in result['results'][0]['context']
        result = json.loads(await execute_personal_tool(202, 'search_documents', {'query': 'summarize', 'document_id': doc_id}))
        assert result['results'] == []
    finally:
        await database.fetch('DELETE FROM documents WHERE id = %s AND telegram_user_id = 101', (doc_id,))


@pytest.mark.asyncio
async def test_reminder_create_edit_and_timezone(database):
    from tools.personal_tools import execute_personal_tool
    await database.set_timezone(101, 'Asia/Yerevan')
    result = json.loads(await execute_personal_tool(101, 'create_reminder', {
        'content': 'Study', 'due_at': '2030-01-01T17:00:00+04:00', 'recurrence': 'weekly'}))
    assert result['success']
    reminder_id = result['results'][0]['id']
    result = json.loads(await execute_personal_tool(101, 'edit_reminder', {
        'reminder_id': reminder_id, 'due_at': '2030-01-02T17:00:00+04:00'}))
    assert result['success']
    result = json.loads(await execute_personal_tool(101, 'list_reminders', {}))
    assert result['results'][0]['id'] == reminder_id
    result = json.loads(await execute_personal_tool(202, 'edit_reminder', {'reminder_id': reminder_id, 'content': 'No'}))
    assert not result['success']
