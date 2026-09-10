import os
from unittest.mock import AsyncMock, Mock

import pytest

os.environ.setdefault('TELEGRAM_TOKEN', 'test-telegram-token')
os.environ.setdefault('GROQ_API_KEY', 'test-groq-key')
os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost:5432/test')

from services import memory


@pytest.fixture
def retrieval(monkeypatch):
    # Importance changes the order of the two best matches. A high-importance
    # candidate below the similarity threshold must still be excluded.
    rows = [(i, f'memory {i}', 'fact', importance, 'explicit_user', None, None)
            for i, importance in [(1, 1), (2, 5), (3, 5), (4, 3)]]
    scores = {1: 0.80, 2: 0.79, 3: 0.34, 4: 0.60}
    listing = AsyncMock(return_value=rows)
    get = AsyncMock(side_effect=lambda user, i: (i, '', '', 3, [scores[i]], None, 'explicit_user', None, None, None))
    touch = AsyncMock(return_value=2)
    query = Mock(return_value=[1.0])
    monkeypatch.setattr(memory, 'list_long_term_memories', listing)
    monkeypatch.setattr(memory, 'get_long_term_memory', get)
    monkeypatch.setattr(memory, 'touch_long_term_memories', touch)
    monkeypatch.setattr(memory, 'generate_query_embedding', query)
    monkeypatch.setattr(memory, 'cosine_similarity', lambda q, e: e[0])
    async def inline_thread(function, *args):
        return function(*args)
    monkeypatch.setattr(memory.asyncio, 'to_thread', inline_thread)
    return listing, touch, query


@pytest.mark.asyncio
async def test_only_final_selection_is_touched(retrieval):
    listing, touch, query = retrieval
    result = await memory.retrieve_relevant_memories(123, ' query ', limit=2)
    assert [m['id'] for m in result] == [2, 1]
    assert result[0] == dict(id=2, content='memory 2', memory_type='fact',
                             importance=5, similarity=0.79, score=pytest.approx(0.84))
    touch.assert_awaited_once_with(123, [2, 1])
    listing.assert_awaited_once_with(123, limit=memory.MEMORY_CANDIDATE_LIMIT)
    query.assert_called_once_with('query')


@pytest.mark.asyncio
async def test_touch_failure_preserves_results(retrieval, caplog):
    _, touch, _ = retrieval
    touch.side_effect = RuntimeError('simulated timestamp failure')
    result = await memory.retrieve_relevant_memories(123, 'query', limit=2)
    assert [m['id'] for m in result] == [2, 1]
    assert 'Failed to update memory access timestamps user_id=123' in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['blank', 'no_candidates', 'no_matches', 'zero_limit', 'embedding_failure'])
async def test_empty_selection_is_not_touched(retrieval, monkeypatch, case):
    listing, touch, query = retrieval
    if case == 'no_candidates':
        listing.return_value = []
    if case == 'no_matches':
        monkeypatch.setattr(memory, 'cosine_similarity', lambda q, e: 0.1)
    if case == 'embedding_failure':
        query.side_effect = RuntimeError('simulated embedding failure')
    result = await memory.retrieve_relevant_memories(
        123, ' ' if case == 'blank' else 'query',
        limit=0 if case == 'zero_limit' else 5)
    assert result == []
    touch.assert_not_awaited()


from datetime import datetime, timedelta, timezone


def test_time_decay_bounds_and_missing_values():
    now = datetime.now(timezone.utc)
    assert memory.memory_time_decay(None, now, 7) == 0
    assert memory.memory_time_decay(now, now, 7) == 1
    assert memory.memory_time_decay(now + timedelta(days=1), now, 7) == 1
    assert memory.memory_time_decay(now - timedelta(days=7), now, 7) == 0.5
    assert memory.memory_time_decay(now.replace(tzinfo=None), now, 7) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('recent_field', [8, 9])
async def test_recent_update_or_use_breaks_similarity_tie(retrieval, monkeypatch, recent_field):
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=365)
    rows = [(i, f'memory {i}', 'fact', 3, 'explicit_user', old, old) for i in [1, 2]]
    retrieval[0].return_value = rows
    def full(user, i):
        row = [i, f'memory {i}', 'fact', 3, [0.8], None, 'explicit_user', old, old, None]
        if i == 2:
            row[recent_field] = now
        return row
    monkeypatch.setattr(memory, 'get_long_term_memory', AsyncMock(side_effect=full))
    result = await memory.retrieve_relevant_memories(123, 'query', limit=1)
    assert [item['id'] for item in result] == [2]
    retrieval[1].assert_awaited_once_with(123, [2])


@pytest.mark.asyncio
async def test_recent_use_never_admits_irrelevant_memory(retrieval, monkeypatch):
    now = datetime.now(timezone.utc)
    retrieval[0].return_value = [(1, 'irrelevant', 'fact', 5, 'explicit_user', now, now)]
    monkeypatch.setattr(memory, 'get_long_term_memory', AsyncMock(return_value=(
        1, 'irrelevant', 'fact', 5, [0.34], None, 'explicit_user', now, now, now)))
    assert await memory.retrieve_relevant_memories(123, 'query') == []
    retrieval[1].assert_not_awaited()
