import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.scheduling import parse_due, next_occurrence
from tools import personal_tools


def test_reminder_requires_future_time_and_matching_zone():
    now = datetime(2026, 9, 10, tzinfo=timezone.utc)
    assert parse_due('2026-09-11T17:00:00+04:00', 'Asia/Yerevan', now).hour == 13
    for value in ['2026-09-11T17:00:00', '2026-09-09T17:00:00+04:00', '2026-09-11T17:00:00+03:00']:
        with pytest.raises(ValueError):
            parse_due(value, 'Asia/Yerevan', now)


def test_calendar_recurrence_keeps_wall_clock_across_dst():
    from zoneinfo import ZoneInfo
    zone = ZoneInfo('America/New_York')
    due = datetime(2026, 3, 7, 9, tzinfo=zone)
    next_due = next_occurrence(due, zone.key, 'daily', due)
    assert next_due.astimezone(zone).hour == 9
    assert (next_due - due.astimezone(timezone.utc)).total_seconds() == 23 * 3600


def test_overdue_recurrence_skips_missed_runs():
    due = datetime(2020, 1, 1, 9, tzinfo=timezone.utc)
    now = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
    result = next_occurrence(due, 'UTC', 'weekly', now)
    assert result > now
    assert result.weekday() == due.weekday()
    assert next_occurrence(due, 'UTC', 'none', now) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('name, args', [
    ('create_reminder', {'content': 'hello', 'due_at': 'bad', 'telegram_user_id': 999}),
    ('edit_task', {'task_id': True}), ('delete_note', {'note_id': '1'}),
    ('edit_task', {'task_id': 1, 'priority': 99}),
])
async def test_invalid_arguments_never_reach_database(monkeypatch, name, args):
    fetch = AsyncMock()
    monkeypatch.setattr(personal_tools, 'fetch', fetch)
    result = json.loads(await personal_tools.execute_personal_tool(1, name, args))
    assert result['success'] is False
    fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_reminder_needs_explicit_timezone(monkeypatch):
    monkeypatch.setattr(personal_tools, 'get_timezone', AsyncMock(return_value=None))
    result = json.loads(await personal_tools.execute_personal_tool(1, 'create_reminder',
        {'content': 'hello', 'due_at': '2030-01-01T12:00:00+00:00'}))
    assert not result['success']
    assert 'timezone' in result['error']


@pytest.mark.asyncio
async def test_web_search_without_key_works(monkeypatch):
    from services import web_search
    monkeypatch.setattr(web_search, 'search_web', AsyncMock(return_value=[{'title': 'Python', 'url': 'https://python.org', 'snippet': 'Official'}]))
    result = json.loads(await personal_tools.execute_personal_tool(1, 'web_search', {'query': 'python'}))
    assert result['success']
    assert result['results'][0]['url'] == 'https://python.org'


@pytest.mark.asyncio
async def test_ai_rejects_non_object_tool_arguments(monkeypatch):
    from services import ai
    message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id='1', function=SimpleNamespace(name='delete_task', arguments='[]'))])
    answer = SimpleNamespace(content='Please provide a valid task.', tool_calls=[])
    create = AsyncMock(side_effect=[SimpleNamespace(choices=[SimpleNamespace(message=message)]), SimpleNamespace(choices=[SimpleNamespace(message=answer)])])
    monkeypatch.setattr(ai.client.chat.completions, 'create', create)
    monkeypatch.setattr(ai, 'get_timezone', AsyncMock(return_value='UTC'))
    monkeypatch.setattr(ai, 'record_tool', AsyncMock())
    execute = AsyncMock()
    monkeypatch.setattr(ai, 'execute_task_tool', execute)
    assert await ai.ask_ai(1, 'delete a task', []) == answer.content
    execute.assert_not_awaited()
    tool_message = create.call_args.kwargs['messages'][-1]
    assert json.loads(tool_message['content'])['success'] is False


@pytest.mark.asyncio
async def test_rag_excludes_irrelevant_tail(monkeypatch):
    from rag import retrieval
    connection = AsyncMock()
    connection.__aenter__.return_value = connection
    cursor = AsyncMock()
    cursor.fetchall.return_value = [(1, 1, 'a.txt', 0, 'python programming', None, None),
                                   (2, 1, 'a.txt', 1, 'irrelevant cooking advice', None, None)]
    connection.execute.return_value = cursor
    monkeypatch.setattr(retrieval.psycopg.AsyncConnection, 'connect', AsyncMock(return_value=connection))
    monkeypatch.setattr(retrieval, 'backfill_missing_embeddings', AsyncMock(return_value={}))
    async def fail_embedding(*args):
        raise RuntimeError('offline')
    monkeypatch.setattr(retrieval.asyncio, 'to_thread', fail_embedding)
    result = await retrieval.get_document_context(1, 'python')
    assert 'python programming' in result
    assert 'cooking' not in result


@pytest.mark.asyncio
async def test_command_menu_and_worker_lifecycle(monkeypatch):
    import asyncio
    from unittest.mock import Mock
    from bot import app
    from database import upgrades
    from services import jobs
    monkeypatch.setattr(app, 'initialize_database', AsyncMock())
    monkeypatch.setattr(app, 'initialize_long_term_memory', AsyncMock())
    monkeypatch.setattr(upgrades, 'initialize_upgrades', AsyncMock())
    async def worker(bot):
        await asyncio.Event().wait()
    monkeypatch.setattr(jobs, 'run_jobs', worker)
    application = SimpleNamespace(bot=AsyncMock(), bot_data={}, add_handler=Mock(), add_error_handler=Mock())
    await app.post_init(application)
    task = application.bot_data['reminder_worker']
    application.bot.set_my_commands.assert_awaited_once()
    assert application.add_handler.call_count == 9
    await app.post_shutdown(application)
    assert task.cancelled()
    assert 'reminder_worker' not in application.bot_data
    await app.post_shutdown(application)


@pytest.mark.asyncio
async def test_guard_private_chat_allowlist_and_rate_limit(monkeypatch):
    import config
    from bot.personal_commands import guard, _recent
    from telegram.ext import ApplicationHandlerStop
    _recent.clear()
    monkeypatch.setattr(config, 'ALLOWED_USER_IDS', {101})
    update = SimpleNamespace(effective_user=SimpleNamespace(id=101), effective_message=object(),
                             effective_chat=SimpleNamespace(type='group'))
    with pytest.raises(ApplicationHandlerStop):
        await guard(update, None)
    update.effective_chat.type = 'private'
    update.effective_user.id = 202
    with pytest.raises(ApplicationHandlerStop):
        await guard(update, None)
    update.effective_user.id = 101
    for _ in range(20):
        await guard(update, None)
    with pytest.raises(ApplicationHandlerStop):
        await guard(update, None)
    _recent.clear()
