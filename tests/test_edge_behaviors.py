import asyncio
import io
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize('arguments', [
    {'title': ' ', 'priority': 5}, {'title': 'Task', 'priority': True},
    {'title': 'Task', 'priority': 6}, {'title': 'Task', 'telegram_user_id': 202},
    {'title': 'Task', 'recurrence': 'weekly', 'due_date': 'tomorrow'},
])
async def test_invalid_task_creation_has_no_side_effect(monkeypatch, arguments):
    from tools import task_tools
    create = AsyncMock()
    monkeypatch.setattr(task_tools, 'create_task', create)
    result = json.loads(await task_tools.execute_task_tool(101, 'create_task', arguments))
    assert result['success'] is False
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_failure_and_blank_query_are_graceful(monkeypatch):
    from services import web_search
    from tools.personal_tools import execute_personal_tool
    web_search._cache.clear()
    def fail(query):
        raise RuntimeError('simulated provider rate limit')
    monkeypatch.setattr(web_search, '_search', fail)
    result = json.loads(await execute_personal_tool(101, 'web_search', {'query': 'latest Python release'}))
    assert not result['success'] and 'temporarily unavailable' in result['error']
    assert 'simulated provider' not in result['error']
    assert not json.loads(await execute_personal_tool(101, 'web_search', {'query': '  '}))['success']


@pytest.mark.asyncio
async def test_search_filters_bad_links_deduplicates_and_caches(monkeypatch):
    from services import web_search
    web_search._cache.clear()
    calls = []
    def search(query):
        calls.append(query)
        return [{'title': 'Docs', 'href': 'https://python.org', 'body': 'Source'},
                {'title': 'Duplicate', 'href': 'https://python.org'},
                {'title': 'Bad', 'href': 'javascript:alert(1)'}]
    monkeypatch.setattr(web_search, '_search', search)
    first = await web_search.search_web('fixture')
    assert len(first) == 1
    assert await web_search.search_web('fixture') == first
    assert calls == ['fixture']


@pytest.mark.asyncio
async def test_search_discards_malformed_result_urls(monkeypatch):
    from services import web_search
    web_search._cache.clear()
    monkeypatch.setattr(web_search, '_search', lambda query: [
        {'href': 'http://[invalid'}, {'href': None},
        {'href': 'https://python.org', 'title': None, 'body': None},
    ])
    results = await web_search.search_web('malformed fixtures')
    assert [item['url'] for item in results] == ['https://python.org']


@pytest.mark.asyncio
async def test_telegram_formatting_does_not_corrupt_literal_underscores():
    from utils.telegram_text import send_long_message
    message = SimpleNamespace(reply_text=AsyncMock())
    text = 'Python __init__ documentation: https://example.com/__init__\n' + ('🎸' * 4500)
    await send_long_message(SimpleNamespace(message=message), text)
    parts = [call.args[0] for call in message.reply_text.await_args_list]
    assert '__init__' in parts[0]
    assert all(len(part.encode('utf-16-le')) // 2 <= 4096 for part in parts)
    assert ''.join(parts).count('🎸') == 4500


@pytest.mark.asyncio
async def test_long_message_is_not_formatted_twice():
    from utils.telegram_text import clean_telegram_text, send_long_message
    message = SimpleNamespace(reply_text=AsyncMock())
    text = '**Bold**\n- First item\n\nLast item'
    await send_long_message(SimpleNamespace(message=message), clean_telegram_text(text))
    assert message.reply_text.await_args.args[0] == 'Bold\n• First item\n\nLast item'


@pytest.mark.asyncio
async def test_local_adapter_text_tools_vision(monkeypatch):
    from services.local_ai import LocalClient
    import services.local_ai as local
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'message': {'content': '', 'tool_calls': [
            {'function': {'name': 'search_tasks', 'arguments': {'query': 'guitar'}}}]}})
    real_client = httpx.AsyncClient
    monkeypatch.setattr(local.httpx, 'AsyncClient', lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))
    client = LocalClient()
    result = await client.create(model='qwen3:4b', messages=[{'role': 'user', 'content': 'Find guitar tasks'}], tools=[{'type': 'function', 'function': {'name': 'search_tasks'}}])
    assert result.choices[0].message.tool_calls[0].function.name == 'search_tasks'
    history = [{'role': 'user', 'content': 'Find guitar tasks'}, result.choices[0].message,
               {'role': 'tool', 'name': 'search_tasks', 'content': '{"success": true}'}]
    await client.create(model='qwen3:4b', messages=history)
    assert requests[1]['messages'][1]['tool_calls'][0]['function']['arguments'] == {'query': 'guitar'}
    assert requests[1]['messages'][2]['tool_name'] == 'search_tasks'
    await client.create(model='qwen3-vl:2b', messages=[{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Describe'}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,YWJj'}}]}])
    assert requests[2]['messages'][0]['images'] == ['YWJj']
    assert requests[2]['stream'] is False and requests[2]['think'] is False


@pytest.mark.asyncio
async def test_local_adapter_unavailable_has_no_cloud_fallback(monkeypatch):
    from services import local_ai
    def offline(request):
        raise httpx.ConnectError('local Ollama not running')
    real_client = httpx.AsyncClient
    monkeypatch.setattr(local_ai.httpx, 'AsyncClient', lambda **kw: real_client(transport=httpx.MockTransport(offline), **kw))
    with pytest.raises(httpx.ConnectError):
        await local_ai.LocalClient().create(model='qwen3:4b', messages=[])


@pytest.mark.asyncio
async def test_voice_routes_preserve_cloud_option(monkeypatch):
    from services import voice
    monkeypatch.setattr(voice, 'AI_PROVIDER', 'local')
    monkeypatch.setattr(voice, 'transcribe_local', lambda data: 'Local fixture')
    assert await voice.transcribe_voice(b'fixture') == 'Local fixture'
    cloud = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(text='Cloud fixture')))))
    monkeypatch.setattr(voice, 'AI_PROVIDER', 'groq')
    monkeypatch.setattr(voice, 'voice_client', cloud)
    assert await voice.transcribe_voice(b'fixture') == 'Cloud fixture'
    cloud.audio.transcriptions.create.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['', RuntimeError('invalid audio')])
async def test_voice_silence_and_decode_error_reply_safely(monkeypatch, outcome):
    from bot import handlers
    from test_feature_flows import telegram, replies
    update, context = telegram(payload=b'not audio')
    update.message.voice = SimpleNamespace(file_id='fixture', file_size=9)
    transcribe = AsyncMock(return_value=outcome if isinstance(outcome, str) else None,
                           side_effect=outcome if isinstance(outcome, Exception) else None)
    monkeypatch.setattr(handlers, 'transcribe_voice', transcribe)
    process = AsyncMock()
    monkeypatch.setattr(handlers, 'process_user_message', process)
    await handlers.handle_voice(update, context)
    process.assert_not_awaited()
    assert "couldn't" in replies(update)


def test_dst_gap_and_fall_back_scheduling():
    from zoneinfo import ZoneInfo
    from services.scheduling import next_occurrence, parse_due
    zone = ZoneInfo('America/New_York')
    before = datetime(2026, 3, 7, 2, 30, tzinfo=zone)
    due = next_occurrence(before, zone.key, 'daily', before)
    assert due.astimezone(zone).hour == 3
    before = datetime(2026, 10, 31, 9, tzinfo=zone)
    due = next_occurrence(before, zone.key, 'daily', before)
    assert due.astimezone(zone).hour == 9
    assert (due - before.astimezone(timezone.utc)).total_seconds() == 25 * 3600
    with pytest.raises(ValueError):
        parse_due('2026-03-08T02:30:00-05:00', zone.key, datetime(2026, 1, 1, tzinfo=timezone.utc))


@pytest.mark.asyncio
async def test_rate_limit_expires_and_callback_is_not_confused_with_bot_author(monkeypatch):
    from bot import personal_commands
    from telegram.ext import ApplicationHandlerStop
    import config
    from test_feature_flows import telegram
    personal_commands._recent.clear()
    monkeypatch.setattr(config, 'ALLOWED_USER_IDS', {101})
    clock = [0]
    monkeypatch.setattr(personal_commands, 'monotonic', lambda: clock[0])
    update, context = telegram(data='panel:tasks')
    update.message.from_user = SimpleNamespace(id=999999, is_bot=True)
    for _ in range(20):
        await personal_commands.guard(update, context)
    with pytest.raises(ApplicationHandlerStop):
        await personal_commands.guard(update, context)
    clock[0] = 61
    await personal_commands.guard(update, context)
    personal_commands._recent.clear()


@pytest.mark.asyncio
async def test_image_history_label_matches_utc(monkeypatch):
    from bot import personal_commands
    from test_feature_flows import telegram, replies
    monkeypatch.setattr(personal_commands, 'fetch', AsyncMock(return_value=[{
        'id': 1, 'filename': 'fixture.jpg',
        'created_at': datetime(2026, 9, 10, 17, tzinfo=timezone(timedelta(hours=4))),
    }]))
    update, context = telegram()
    await personal_commands.images_command(update, context)
    assert '13:00 UTC' in replies(update)


@pytest.mark.skipif(__import__('os').getenv('BOT_TEST_LOCAL_WHISPER') != '1',
                    reason='Opt in only with the local base Whisper model already cached')
def test_installed_whisper_silence_and_invalid_audio(monkeypatch):
    import wave
    from services import voice
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setattr(voice, 'VOICE_MODEL', 'base')
    monkeypatch.setattr(voice, 'VOICE_LANGUAGE', 'en')
    voice.local_whisper.cache_clear()
    audio = io.BytesIO()
    with wave.open(audio, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b'\0' * 16000 * 2 * 3)
    try:
        assert voice.transcribe_local(audio.getvalue()) == ''
        from av.error import InvalidDataError
        with pytest.raises(InvalidDataError):
            voice.transcribe_local(b'not audio')
    finally:
        voice.local_whisper.cache_clear()
