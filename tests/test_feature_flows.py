"""User-requested feature flows with a real disposable DB and simulated Telegram.

AI dispatch tests use deterministic model responses: they verify the application
pipeline, not a provider model's natural-language reasoning quality.
"""
import asyncio
import io
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


def telegram(args=(), data=None, payload=b''):
    message = SimpleNamespace(reply_text=AsyncMock(), text='', caption=None)
    update = SimpleNamespace(message=message, effective_message=message,
        effective_user=SimpleNamespace(id=101), effective_chat=SimpleNamespace(id=101, type='private'))
    file = SimpleNamespace(download_as_bytearray=AsyncMock(return_value=bytearray(payload)))
    context = SimpleNamespace(args=list(args), bot=SimpleNamespace(get_file=AsyncMock(return_value=file),
        send_chat_action=AsyncMock()))
    if data:
        update.callback_query = SimpleNamespace(data=data, answer=AsyncMock())
    return update, context


def replies(update):
    return '\n'.join(call.args[0] for call in update.message.reply_text.await_args_list)


async def personal(name, **args):
    from tools.personal_tools import execute_personal_tool
    return json.loads(await execute_personal_tool(101, name, args))


@pytest.mark.asyncio
async def test_advanced_guitar_task_lifecycle(isolated_db):
    from tools.task_tools import execute_task_tool
    from bot.commands import tasks_command
    due = (datetime.now(timezone(timedelta(hours=4))) + timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
    created = json.loads(await execute_task_tool(101, 'create_task', {
        'title': 'Order Sitka spruce', 'priority': 5, 'project': 'guitar',
        'due_date': due.isoformat(), 'notes': 'Solid top, quarter sawn'}))
    assert created['success']
    task_id = created['task_id']
    row = (await isolated_db('SELECT * FROM tasks WHERE id = %s', (task_id,)))[0]
    assert (row['priority'], row['project'], row['due_date']) == (5, 'guitar', due.isoformat())
    update, context = telegram()
    await tasks_command(update, context)
    assert 'Priority 5/5' in replies(update) and 'guitar' in replies(update)
    assert (await personal('edit_task', task_id=task_id, priority=3, due_date=due.replace(hour=20).isoformat()))['success']
    assert (await personal('search_tasks', query='guitar'))['results'][0]['id'] == task_id
    assert (await personal('search_tasks', project='GUITAR'))['results'][0]['id'] == task_id
    assert (await personal('search_tasks', query='quarter sawn'))['results'][0]['id'] == task_id
    assert (await personal('edit_task', task_id=task_id, title='Order solid Sitka top'))['success']
    assert json.loads(await execute_task_tool(101, 'complete_task', {'task_id': task_id}))['success']
    row = (await isolated_db('SELECT * FROM tasks WHERE id = %s', (task_id,)))[0]
    assert (row['status'], row['priority'], row['due_date']) == ('done', 3, due.replace(hour=20).isoformat())
    assert json.loads(await execute_task_tool(101, 'delete_task', {'task_id': task_id}))['success']
    assert not await isolated_db('SELECT id FROM tasks WHERE id = %s', (task_id,))


@pytest.mark.asyncio
async def test_project_note_edit_preserves_project(isolated_db):
    from bot.personal_commands import notes_command
    created = await personal('save_note', title='Guitar top', content='Use solid Sitka spruce', project='guitar')
    note_id = created['results'][0]['id']
    assert (await personal('search_notes', query='guitar'))['results']
    assert (await personal('save_note', note_id=note_id, title='Top material', content='Use quarter-sawn Sitka spruce'))['success']
    row = (await isolated_db('SELECT * FROM knowledge_notes WHERE id = %s', (note_id,)))[0]
    assert row['project'] == 'guitar'
    update, context = telegram(['Sitka'])
    await notes_command(update, context)
    assert 'quarter-sawn Sitka' in replies(update)
    assert (await personal('delete_note', note_id=note_id))['success']
    assert not (await personal('search_notes', query='Sitka'))['results']


def document_bytes(kind):
    content = 'Guitar project: solid Sitka spruce top. Finish with shellac. ' * 4
    if kind == 'txt':
        return content.encode()
    if kind == 'docx':
        from docx import Document
        doc = Document()
        doc.add_paragraph(content)
        out = io.BytesIO()
        doc.save(out)
        return out.getvalue()
    import pymupdf
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 500, 500), content)
        return doc.tobytes()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['txt', 'pdf', 'docx'])
async def test_document_upload_collections_search_delete(isolated_db, monkeypatch, kind):
    from bot import document_handler, commands
    from rag import retrieval
    from config import EMBEDDING_DIMENSIONS
    vector = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)
    monkeypatch.setattr(document_handler, 'generate_passage_embeddings', lambda chunks: [vector for _ in chunks])
    monkeypatch.setattr(retrieval, 'generate_query_embedding', lambda query: vector)
    payload = document_bytes(kind)
    update, context = telegram(payload=payload)
    update.message.document = SimpleNamespace(file_name=f'guitar.{kind}', file_size=len(payload), file_id='sample')
    await document_handler.handle_document(update, context)
    assert 'successfully' in replies(update)
    rows = (await personal('list_documents'))['results']
    doc_id = rows[0]['id']
    assert (await personal('set_document_collection', document_id=doc_id, collection='guitar'))['success']
    for filters in ({}, {'document_id': doc_id}, {'collection': 'guitar'}):
        result = await personal('search_documents', query='Sitka', **filters)
        assert 'Sitka spruce' in result['results'][0]['context']
        assert 'chunk 1' in result['results'][0]['context']
    assert not (await personal('search_documents', query='Sitka', collection='other'))['results']
    await commands.files_command(update, context)
    assert f'guitar.{kind}' in replies(update)
    await commands.clear_files(update, context)
    assert not (await personal('list_documents'))['results']
    assert not await isolated_db('SELECT id FROM document_chunks')


@pytest.mark.asyncio
async def test_image_upload_history_and_commands(isolated_db, monkeypatch):
    from PIL import Image
    from services import vision
    from bot import personal_commands, handlers, commands
    from database.images import load_latest_image
    out = io.BytesIO()
    Image.new('RGB', (50, 50), 'blue').save(out, format='PNG')
    update, context = telegram(payload=out.getvalue())
    fake_vision = AsyncMock(return_value='A blue square.')
    monkeypatch.setattr(vision, 'analyze_image_with_vision', fake_vision)
    monkeypatch.setattr(personal_commands, 'analyze_image_with_vision', fake_vision)
    monkeypatch.setattr(handlers, 'analyze_image_with_vision', fake_vision)
    monkeypatch.setattr(vision, 'ocr_image', lambda image: '')
    for i in range(12):
        await vision.process_image_upload(update, context, 'sample', f'{i}.png', len(out.getvalue()), None)
    fake_vision.assert_not_awaited()
    rows = await isolated_db('SELECT id, filename FROM image_history ORDER BY id')
    assert len(rows) == 10 and rows[0]['filename'] == '2.png'
    await personal_commands.images_command(update, context)
    assert '11.png' in replies(update)
    context.args = [str(rows[0]['id']), 'What', 'color?']
    await personal_commands.image_command(update, context)
    assert 'A blue square.' in replies(update)
    await handlers.process_user_message(update, context, 'What is in this image?')
    assert fake_vision.await_count == 2
    await commands.clear_image_command(update, context)
    assert await load_latest_image(101) is None
    assert len(await isolated_db('SELECT id FROM image_history')) == 10
    context.args = []
    await personal_commands.clear_images_command(update, context)
    assert len(await isolated_db('SELECT id FROM image_history')) == 10
    context.args = ['confirm']
    await personal_commands.clear_images_command(update, context)
    assert not await isolated_db('SELECT id FROM image_history')


@pytest.mark.asyncio
@pytest.mark.parametrize('panel', ['tasks', 'reminders', 'memory', 'notes', 'images'])
async def test_all_inline_panels(isolated_db, panel):
    from bot.personal_commands import panel_callback
    update, context = telegram(data=f'panel:{panel}')
    await panel_callback(update, context)
    update.callback_query.answer.assert_awaited_once()
    assert replies(update)


@pytest.mark.asyncio
async def test_activity_api_is_authorized_and_contains_only_metrics(isolated_db):
    from api import app
    from fastapi.testclient import TestClient
    from database.personal import record_tool
    await personal('save_note', title='Private fixture', content='PRIVATE-CONTENT-MUST-NOT-APPEAR')
    await record_tool(101, 'save_note', True, 40)
    await record_tool(101, 'web_search', False, 80)
    with TestClient(app) as client:
        assert client.get('/activity').status_code == 401
        assert client.get('/activity', headers={'X-API-Key': 'wrong'}).status_code == 401
        result = client.get('/activity', headers={'X-API-Key': 'test-admin-key'})
    assert result.status_code == 200
    assert 'PRIVATE-CONTENT' not in result.text and 'test-admin-key' not in result.text
    tools = {row['tool']: row for row in result.json()['tools_last_24_hours']}
    assert tools['save_note']['calls'] == 1 and tools['web_search']['failures'] == 1


@pytest.mark.asyncio
async def test_reminder_worker_terminal_errors_and_retry_after(isolated_db):
    from services.jobs import deliver_one
    from telegram.error import Forbidden, RetryAfter
    now = datetime.now(timezone.utc)
    for error, expected in [(Forbidden('blocked'), 'failed'), (RetryAfter(60), 'pending')]:
        job = (await isolated_db("INSERT INTO reminders (telegram_user_id, content, due_at, timezone) VALUES (101, 'Fixture', %s, 'UTC') RETURNING id", (now - timedelta(minutes=1),)))[0]
        bot = SimpleNamespace(send_message=AsyncMock(side_effect=error))
        assert await deliver_one(bot)
        row = (await isolated_db('SELECT * FROM reminders WHERE id = %s', (job['id'],)))[0]
        assert row['status'] == expected and row['last_error'] == type(error).__name__
        if expected == 'pending':
            assert row['retry_at'] >= now + timedelta(seconds=59)


@pytest.mark.asyncio
async def test_worker_cancellation_rolls_back_claim(isolated_db):
    from services.jobs import deliver_one
    await isolated_db("INSERT INTO reminders (telegram_user_id, content, due_at, timezone) VALUES (101, 'Fixture', NOW(), 'UTC')")
    sending = asyncio.Event()
    async def paused_send(**kwargs):
        sending.set()
        await asyncio.Event().wait()
    worker = asyncio.create_task(deliver_one(SimpleNamespace(send_message=paused_send)))
    await asyncio.wait_for(sending.wait(), 5)
    worker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await worker
    assert (await isolated_db('SELECT status, attempts FROM reminders'))[0] == {'status': 'pending', 'attempts': 0}


@pytest.mark.asyncio
async def test_memory_dedup_replacement_and_access(isolated_db, monkeypatch):
    from config import EMBEDDING_DIMENSIONS
    from services import memory
    from database.long_term_memory import get_long_term_memory, list_long_term_memories
    a = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)
    b = [0.0, 1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 2)
    monkeypatch.setattr(memory, 'generate_passage_embeddings', lambda texts: [b if 'detailed' in text else a for text in texts])
    monkeypatch.setattr(memory, 'generate_query_embedding', lambda query: a)
    one = await memory.remember_user_memory(101, 'I prefer concise answers', memory_type='preference', importance=5)
    assert await memory.remember_user_memory(101, 'I prefer short answers') == one
    assert len(await list_long_term_memories(101)) == 1
    assert (await get_long_term_memory(101, one))[9] is None
    assert (await memory.retrieve_relevant_memories(101, 'answer style'))[0]['id'] == one
    assert (await get_long_term_memory(101, one))[9] is not None
    assert await memory.replace_user_memory(101, one, 'I prefer detailed answers', 'preference') == one
    assert (await get_long_term_memory(101, one))[1] == 'I prefer detailed answers'
    assert await memory.replace_user_memory(202, one, 'Wrong user', 'fact') is None


@pytest.mark.asyncio
async def test_help_exposes_all_five_panels():
    from bot.personal_commands import help_command
    update, context = telegram()
    await help_command(update, context)
    assert all(command in replies(update) for command in ['/tasks', '/reminders', '/notes', '/images', '/image', '/timezone'])
    keyboard = update.message.reply_text.await_args.kwargs['reply_markup'].inline_keyboard
    assert {button.callback_data for row in keyboard for button in row} == {
        'panel:tasks', 'panel:reminders', 'panel:memory', 'panel:notes', 'panel:images'}


@pytest.mark.asyncio
async def test_daily_task_creates_one_next_occurrence_with_metadata(isolated_db):
    from tools.task_tools import execute_task_tool
    from database.tasks import complete_task
    due = (datetime.now(timezone.utc) + timedelta(days=1)).date()
    result = json.loads(await execute_task_tool(101, 'create_task', {
        'title': 'Practice guitar', 'due_date': due.isoformat(), 'recurrence': 'daily',
        'priority': 5, 'project': 'guitar', 'notes': 'Scales'}))
    assert await complete_task(101, result['task_id']) == 'Practice guitar'
    assert await complete_task(101, result['task_id']) is None
    rows = await isolated_db("SELECT * FROM tasks WHERE status = 'open'")
    assert len(rows) == 1
    assert rows[0]['due_date'] == (due + timedelta(days=1)).isoformat()
    assert (rows[0]['priority'], rows[0]['project'], rows[0]['notes']) == (5, 'guitar', 'Scales')


@pytest.mark.asyncio
async def test_ai_task_dispatch_persists_arguments_and_records_activity(isolated_db, monkeypatch):
    from services import ai
    from database.personal import set_timezone
    await set_timezone(101, 'Asia/Yerevan')
    message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id='fixture-call',
        function=SimpleNamespace(name='create_task', arguments=json.dumps({
            'title': 'Order Sitka spruce', 'priority': 5, 'project': 'guitar',
            'due_date': '2030-01-01T18:00:00+04:00'})))])
    answer = SimpleNamespace(content='Task saved.', tool_calls=[])
    create = AsyncMock(side_effect=[SimpleNamespace(choices=[SimpleNamespace(message=message)]),
                                   SimpleNamespace(choices=[SimpleNamespace(message=answer)])])
    monkeypatch.setattr(ai.client.chat.completions, 'create', create)
    assert await ai.ask_ai(101, 'Create a high priority guitar task to order Sitka spruce', []) == 'Task saved.'
    row = (await isolated_db('SELECT * FROM tasks'))[0]
    assert (row['priority'], row['project']) == (5, 'guitar')
    result = create.call_args.kwargs['messages'][-1]
    assert result['role'] == 'tool' and json.loads(result['content'])['success']
    events = await isolated_db('SELECT tool, success FROM tool_events')
    assert events == [{'tool': 'create_task', 'success': True}]
