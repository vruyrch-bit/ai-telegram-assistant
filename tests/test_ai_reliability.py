import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import groq
import httpx
import pytest

from services.ai_requests import AIRequestError, request_completion
from services.ai_responses import final_answer


def completion(content='', calls=None, finish_reason='stop'):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=calls or []),
        finish_reason=finish_reason)])


def api_error(status=429, headers=None, code=None):
    response = httpx.Response(status, headers=headers,
                              request=httpx.Request('POST', 'https://example.test/chat'))
    cls = {429: groq.RateLimitError, 503: groq.InternalServerError,
           400: groq.BadRequestError, 401: groq.AuthenticationError}[status]
    return cls('private provider payload', response=response, body={'code': code})


@pytest.mark.asyncio
async def test_completion_retries_short_cooldown_once(monkeypatch):
    from services import ai_requests
    sleep = AsyncMock()
    monkeypatch.setattr(ai_requests.asyncio, 'sleep', sleep)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=[api_error(headers={'retry-after-ms': '250'}), completion('Ready')]))))
    assert (await request_completion(client, model='test', messages=[])).choices[0].message.content == 'Ready'
    assert client.chat.completions.create.await_count == 2
    sleep.assert_awaited_once_with(0.25)


@pytest.mark.asyncio
async def test_long_cooldown_does_not_retry_early_or_expose_payload(monkeypatch, caplog):
    from services import ai_requests
    sleep = AsyncMock()
    monkeypatch.setattr(ai_requests.asyncio, 'sleep', sleep)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=api_error(headers={'retry-after': '45'})))))
    with pytest.raises(AIRequestError, match='rate limit') as error:
        await request_completion(client, model='test', messages=[])
    assert 'private' not in str(error.value)
    assert 'private' not in caplog.text
    client.chat.completions.create.assert_awaited_once()
    sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [429, 503])
async def test_transient_failures_stop_after_one_retry(monkeypatch, status):
    from services import ai_requests
    monkeypatch.setattr(ai_requests.asyncio, 'sleep', AsyncMock())
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=api_error(status)))))
    with pytest.raises(AIRequestError):
        await request_completion(client, model='test', messages=[])
    assert client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [400, 401])
async def test_configuration_errors_are_not_retried(status):
    original = api_error(status)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=original))))
    with pytest.raises(type(original)):
        await request_completion(client, model='test', messages=[])
    client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancellation_propagates():
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=asyncio.CancelledError))))
    with pytest.raises(asyncio.CancelledError):
        await request_completion(client, model='test', messages=[])


@pytest.mark.parametrize('raw, expected', [
    ('<think>private analysis</think>Blue circle.', 'Blue circle.'),
    ('<THINK>private\nanalysis</THINK>Red square.', 'Red square.'),
    ('<think>unfinished private analysis', ''),
    ('private analysis</think>Final answer.', 'Final answer.'),
    ('__init__ https://example.com/some_path', '__init__ https://example.com/some_path'),
])
def test_only_final_answer_is_kept(raw, expected):
    assert final_answer(raw) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('raw, reason, expected', [
    ('<think>private</think>Blue circle.', 'stop', 'Blue circle.'),
    ('<think>private unfinished', 'length', "I couldn't finish the image answer."),
    ('Blue circle and', 'length', 'The answer was cut short.'),
])
async def test_vision_hides_reasoning_and_handles_truncation(monkeypatch, raw, reason, expected):
    from services import vision
    create = AsyncMock(return_value=completion(raw, finish_reason=reason))
    monkeypatch.setattr(vision.vision_client.chat.completions, 'create', create)
    monkeypatch.setattr(vision, 'AI_PROVIDER', 'groq')
    monkeypatch.setattr(vision, 'VISION_MODEL', 'qwen/qwen3.6-27b')
    result = await vision.analyze_image_with_vision(b'fixture', 'What shape?')
    assert expected in result
    assert 'private' not in result and '<think>' not in result
    assert create.await_args.kwargs['reasoning_format'] == 'hidden'
    assert create.await_args.kwargs['reasoning_effort'] == 'none'


@pytest.mark.asyncio
@pytest.mark.parametrize('succeeded', [True, False])
async def test_completed_action_reported_when_final_ai_call_fails(monkeypatch, succeeded):
    from services import ai
    call = SimpleNamespace(id='reminder-call', function=SimpleNamespace(
        name='create_reminder', arguments='{"content":"Fixture"}'))
    create = AsyncMock(side_effect=[completion(calls=[call]),
                                   api_error(headers={'retry-after': '45'})])
    monkeypatch.setattr(ai.client.chat.completions, 'create', create)
    monkeypatch.setattr(ai, 'get_timezone', AsyncMock(return_value='UTC'))
    monkeypatch.setattr(ai, 'record_tool', AsyncMock())
    execute = AsyncMock(return_value=json.dumps({'success': succeeded, 'results': [{'id': 123}]}))
    monkeypatch.setattr(ai, 'execute_personal_tool', execute)
    result = await ai.ask_ai(101, 'Remind me to check the fixture.', [])
    assert 'rate limit' in result
    assert ('Reminder saved' in result) is succeeded
    execute.assert_awaited_once()
    assert create.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('nested', [False, True])
async def test_search_sources_survive_invalid_followup_tool_call(monkeypatch, nested):
    from services import ai
    call = SimpleNamespace(id='search-call', function=SimpleNamespace(
        name='web_search', arguments='{"query":"Python release"}'))
    error = api_error(400, code='tool_use_failed')
    if nested:
        error.body = {'error': error.body}
    create = AsyncMock(side_effect=[completion(calls=[call]), error])
    monkeypatch.setattr(ai.client.chat.completions, 'create', create)
    monkeypatch.setattr(ai, 'get_timezone', AsyncMock(return_value='UTC'))
    monkeypatch.setattr(ai, 'record_tool', AsyncMock())
    monkeypatch.setattr(ai, 'execute_personal_tool', AsyncMock(return_value=json.dumps({
        'success': True, 'results': [{'title': 'Official Python', 'url': 'https://python.org'}]})))
    result = await ai.ask_ai(101, 'Search the web for Python.', [])
    assert 'https://python.org' in result
    assert 'private' not in result
    assert 'couldn\'t form a valid action' in result


@pytest.mark.asyncio
async def test_real_sdk_error_body_is_handled_without_replaying_tools():
    requests = []

    async def respond(request):
        requests.append(request)
        return httpx.Response(400, json={'error': {
            'code': 'tool_use_failed', 'message': 'private provider payload',
            'failed_generation': 'private tool arguments',
        }})

    async with groq.AsyncGroq(api_key='test-key', max_retries=0,
                             http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))) as client:
        with pytest.raises(AIRequestError, match='valid action') as error:
            await request_completion(client, model='test', messages=[{'role': 'user', 'content': 'Test'}])
    assert len(requests) == 1
    assert 'private' not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize('final_response', ['empty', 'unauthorized'])
async def test_completed_actions_survive_other_incomplete_responses(monkeypatch, final_response):
    from services import ai
    call = SimpleNamespace(id='reminder-call', function=SimpleNamespace(
        name='create_reminder', arguments='{"content":"Fixture"}'))
    final = completion() if final_response == 'empty' else api_error(401)
    monkeypatch.setattr(ai.client.chat.completions, 'create', AsyncMock(side_effect=[completion(calls=[call]), final]))
    monkeypatch.setattr(ai, 'get_timezone', AsyncMock(return_value='UTC'))
    monkeypatch.setattr(ai, 'record_tool', AsyncMock())
    execute = AsyncMock(return_value=json.dumps({'success': True, 'results': [{'id': 123}]}))
    monkeypatch.setattr(ai, 'execute_personal_tool', execute)
    result = await ai.ask_ai(101, 'Remind me to check the fixture.', [])
    assert 'Reminder saved' in result and 'private' not in result
    execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_old_reasoning_not_resent_and_history_not_modified(monkeypatch):
    from services import ai
    create = AsyncMock(return_value=completion('Ready'))
    monkeypatch.setattr(ai.client.chat.completions, 'create', create)
    monkeypatch.setattr(ai, 'get_timezone', AsyncMock(return_value='UTC'))
    history = [{'role': 'assistant', 'content': '<think>private</think>Blue circle.'}]
    await ai.ask_ai(101, 'Hello', history)
    assert {'role': 'assistant', 'content': 'Blue circle.'} in create.await_args.kwargs['messages']
    assert 'private' in history[0]['content']


@pytest.mark.asyncio
async def test_explicit_image_command_uses_friendly_ai_error():
    from bot.personal_commands import on_error
    from telegram import Chat, Message, Update
    # Use the error-handler's supported Telegram Update path.
    update = Update(1, message=Message(1, None, Chat(101, 'private'), text='/image 1 question'))
    from unittest.mock import patch
    with patch.object(Message, 'reply_text', new_callable=AsyncMock) as reply:
        await on_error(update, SimpleNamespace(error=AIRequestError('The AI is temporarily busy.')))
    reply.assert_awaited_once_with('The AI is temporarily busy.')


def test_ddgs_uses_available_backends_automatically(monkeypatch):
    from services import web_search
    search = Mock()
    search.text.return_value = [{'title': 'Source', 'href': 'https://python.org'}]
    factory = Mock()
    factory.return_value.__enter__ = Mock(return_value=search)
    factory.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(web_search, 'DDGS', factory)
    assert web_search._search('fixture')[0]['href'] == 'https://python.org'
    search.text.assert_called_once_with('fixture', max_results=5, backend='auto', safesearch='moderate')


@pytest.mark.asyncio
async def test_groq_rate_limit_falls_back_to_openrouter(
    monkeypatch,
):
    import services.ai_requests as ai_requests

    class FakeRateLimitError(
        groq.APIError
    ):
        pass

    class FakeCompletions:
        async def create(
            self,
            **kwargs,
        ):
            error = FakeRateLimitError(
                "rate limited",
                request=None,
                body={},
            )
            error.status_code = 429
            error.response = None
            raise error

    class FakeClient:
        class Chat:
            completions = FakeCompletions()

        chat = Chat()

    class FakeFallbackResponse:
        pass

    fallback_response = (
        FakeFallbackResponse()
    )

    async def fake_openrouter(
        **kwargs,
    ):
        return fallback_response

    monkeypatch.setattr(
        ai_requests,
        "openrouter_is_configured",
        lambda: True,
    )

    monkeypatch.setattr(
        ai_requests,
        "request_openrouter_completion",
        fake_openrouter,
    )

    monkeypatch.setattr(
        ai_requests,
        "_retry_delay",
        lambda error: 10,
    )

    response = (
        await ai_requests.request_completion(
            FakeClient(),
            model="primary-model",
            messages=[
                {
                    "role": "user",
                    "content": "hello",
                }
            ],
        )
    )

    assert response is fallback_response


@pytest.mark.asyncio
async def test_groq_rate_limit_without_fallback_returns_safe_error(
    monkeypatch,
):
    import services.ai_requests as ai_requests

    class FakeRateLimitError(
        groq.APIError
    ):
        pass

    class FakeCompletions:
        async def create(
            self,
            **kwargs,
        ):
            error = FakeRateLimitError(
                "rate limited",
                request=None,
                body={},
            )
            error.status_code = 429
            error.response = None
            raise error

    class FakeClient:
        class Chat:
            completions = FakeCompletions()

        chat = Chat()

    monkeypatch.setattr(
        ai_requests,
        "openrouter_is_configured",
        lambda: False,
    )

    monkeypatch.setattr(
        ai_requests,
        "_retry_delay",
        lambda error: 10,
    )

    with pytest.raises(
        ai_requests.AIRequestError
    ) as exc_info:
        await ai_requests.request_completion(
            FakeClient(),
            model="primary-model",
            messages=[
                {
                    "role": "user",
                    "content": "hello",
                }
            ],
        )

    assert (
        "rate limit"
        in str(
            exc_info.value
        ).lower()
    )


@pytest.mark.asyncio
async def test_openrouter_remains_sticky_after_fallback(
    monkeypatch,
):
    import services.ai_requests as ai_requests

    calls = {
        "groq": 0,
        "openrouter": 0,
    }

    class FakeGroqCompletions:
        async def create(
            self,
            **kwargs,
        ):
            calls["groq"] += 1

            error = groq.APIStatusError(
                "rate limited",
                response=type(
                    "Response",
                    (),
                    {
                        "status_code": 429,
                        "headers": {
                            "retry-after": "10",
                        },
                        "request": None,
                    },
                )(),
                body={},
            )

            raise error

    class FakeGroqClient:
        class Chat:
            completions = (
                FakeGroqCompletions()
            )

        chat = Chat()

    responses = [
        object(),
        object(),
    ]

    async def fake_openrouter(
        **kwargs,
    ):
        calls["openrouter"] += 1
        return responses[
            calls["openrouter"] - 1
        ]

    monkeypatch.setattr(
        ai_requests,
        "openrouter_is_configured",
        lambda: True,
    )

    monkeypatch.setattr(
        ai_requests,
        "request_openrouter_completion",
        fake_openrouter,
    )

    provider_state = {}

    first = (
        await ai_requests.request_completion(
            FakeGroqClient(),
            provider_state=provider_state,
            model="groq-model",
            messages=[],
        )
    )

    assert first is responses[0]
    assert (
        provider_state["provider"]
        == "openrouter"
    )

    second = (
        await ai_requests.request_completion(
            FakeGroqClient(),
            provider_state=provider_state,
            model="groq-model",
            messages=[],
        )
    )

    assert second is responses[1]

    # Groq was tried only before fallback.
    assert calls["groq"] == 1

    # Both remaining completions used
    # OpenRouter.
    assert calls["openrouter"] == 2
