"""Exercise real request serialization with mocked transports, never live keys."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import groq
import httpx
from openai import AsyncOpenAI
import pytest

from services import ai_requests, cloud_fallback


def rate_limit():
    response = httpx.Response(429, headers={'retry-after': '45'},
                              request=httpx.Request('POST', 'https://example.test/chat'))
    return groq.RateLimitError('private payload', response=response, body={})


def primary(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.mark.asyncio
@pytest.mark.parametrize('vision', [False, True])
async def test_fallback_serializes_text_tools_and_groq_vision_options(monkeypatch, vision):
    captured = []
    async def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={'id': 'fixture', 'object': 'chat.completion',
            'created': 0, 'model': 'fixture:free', 'choices': [{'index': 0,
            'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Blue circle.'}}]})
    client = AsyncOpenAI(api_key='test-only', base_url='https://example.test/v1',
                         max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(cloud_fallback, 'openrouter_client', client)
    monkeypatch.setattr(cloud_fallback, 'OPENROUTER_API_KEY', 'test-only')
    monkeypatch.setattr(cloud_fallback, 'OPENROUTER_MODEL', 'openrouter/free')
    messages = [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'What shape?'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,fixture'}},
    ] if vision else 'List my fixture notes.'}]
    options = ({'temperature': 0.2, 'reasoning_format': 'hidden', 'reasoning_effort': 'none'}
               if vision else {'tools': [{'type': 'function', 'function': {
                   'name': 'search_notes', 'parameters': {'type': 'object', 'properties': {}}}}],
                   'tool_choice': 'auto'})
    create = AsyncMock(side_effect=rate_limit())
    try:
        result = await ai_requests.request_completion(primary(create), model='groq-only',
            messages=messages, max_completion_tokens=1200, **options)
    finally:
        await client.close()
    assert result.choices[0].message.content == 'Blue circle.'
    create.assert_awaited_once()
    assert len(captured) == 1
    body = captured[0]
    assert body['messages'] == messages
    assert body['model'] == 'openrouter/free' and body['max_tokens'] == 1200
    assert body['provider']['require_parameters'] is True
    assert set(body['provider']['max_price'].values()) == {0}
    assert body['reasoning'] == {'exclude': True}
    assert 'reasoning_format' not in body and 'reasoning_effort' not in body
    if vision:
        assert body['temperature'] == 0.2
    else:
        assert body['tools'] == options['tools'] and body['tool_choice'] == 'auto'


@pytest.mark.asyncio
@pytest.mark.parametrize('model', ['openrouter/auto', 'provider/paid', '', 'provider/model:free:online'])
async def test_paid_or_invalid_model_cannot_trigger_network_request(monkeypatch, model):
    create = AsyncMock()
    monkeypatch.setattr(cloud_fallback, 'OPENROUTER_API_KEY', 'test-only')
    monkeypatch.setattr(cloud_fallback, 'OPENROUTER_MODEL', model)
    monkeypatch.setattr(cloud_fallback, 'openrouter_client', primary(create))
    assert not cloud_fallback.openrouter_is_configured()
    assert cloud_fallback.build_openrouter_client() is None
    assert await cloud_fallback.request_openrouter_completion(messages=[]) is None
    create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('response', [None, SimpleNamespace(choices=[]),
    SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='', tool_calls=[]))])])
async def test_empty_fallback_keeps_friendly_error(monkeypatch, response):
    monkeypatch.setattr(cloud_fallback, 'OPENROUTER_API_KEY', 'test-only')
    monkeypatch.setattr(cloud_fallback, 'openrouter_client', primary(AsyncMock(return_value=response)))
    with pytest.raises(ai_requests.AIRequestError, match='rate limit'):
        await ai_requests.request_completion(primary(AsyncMock(side_effect=rate_limit())),
                                              model='fixture', messages=[])


@pytest.mark.asyncio
async def test_fallback_failure_is_bounded_and_does_not_leak_payload(monkeypatch, caplog):
    fallback = AsyncMock(side_effect=RuntimeError('private provider payload and secret'))
    monkeypatch.setattr(ai_requests, 'openrouter_is_configured', lambda: True)
    monkeypatch.setattr(ai_requests, 'request_openrouter_completion', fallback)
    with pytest.raises(ai_requests.AIRequestError, match='rate limit') as exc:
        await ai_requests.request_completion(primary(AsyncMock(side_effect=rate_limit())),
                                              model='fixture', messages=[])
    fallback.assert_awaited_once()
    assert 'private' not in caplog.text and 'secret' not in str(exc.value)


@pytest.mark.asyncio
async def test_cancelled_fallback_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(ai_requests, 'openrouter_is_configured', lambda: True)
    monkeypatch.setattr(ai_requests, 'request_openrouter_completion', AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await ai_requests.request_completion(primary(AsyncMock(side_effect=rate_limit())),
                                              model='fixture', messages=[])


@pytest.mark.asyncio
async def test_successful_groq_does_not_send_to_fallback(monkeypatch):
    fallback = AsyncMock()
    monkeypatch.setattr(ai_requests, 'openrouter_is_configured', lambda: True)
    monkeypatch.setattr(ai_requests, 'request_openrouter_completion', fallback)
    response = object()
    assert await ai_requests.request_completion(primary(AsyncMock(return_value=response)),
                                                model='fixture', messages=[]) is response
    fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_fallback_continues_completed_action_without_replaying_it(monkeypatch):
    from services import ai
    call = SimpleNamespace(id='fixture-call', function=SimpleNamespace(
        name='create_reminder', arguments='{"content":"Fixture"}'))
    tool_response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=None, tool_calls=[call]))])
    final_response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content='Reminder saved.', tool_calls=[]))])
    create = AsyncMock(side_effect=[tool_response, rate_limit()])
    fallback = AsyncMock(return_value=final_response)
    execute = AsyncMock(return_value=json.dumps({'success': True, 'results': [{'id': 321}]}))
    monkeypatch.setattr(ai.client.chat.completions, 'create', create)
    monkeypatch.setattr(ai, 'get_timezone', AsyncMock(return_value='UTC'))
    monkeypatch.setattr(ai, 'record_tool', AsyncMock())
    monkeypatch.setattr(ai, 'execute_personal_tool', execute)
    monkeypatch.setattr(ai_requests, 'openrouter_is_configured', lambda: True)
    monkeypatch.setattr(ai_requests, 'request_openrouter_completion', fallback)
    assert await ai.ask_ai(101, 'Remind me to check the fixture.', []) == 'Reminder saved.'
    execute.assert_awaited_once()
    fallback.assert_awaited_once()
    last_message = fallback.await_args.kwargs['messages'][-1]
    assert last_message['role'] == 'tool' and last_message['tool_call_id'] == 'fixture-call'
    assert json.loads(last_message['content'])['results'][0]['id'] == 321
