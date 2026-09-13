"""Regressions reproduced during Telegram smoke testing."""
import json

import pytest

from services.vision import should_use_latest_image


@pytest.mark.parametrize('message', [
    'What color and shape are in the latest image? Answer briefly.',
    'Describe my last uploaded photo.',
    'Read the most recent screenshot.',
    'WHAT IS IN MY LATEST PICTURE?',
])
def test_latest_image_wording_routes_to_vision(message):
    assert should_use_latest_image(message)


@pytest.mark.parametrize('message', [
    'What is the latest Python release?',
    'Find my last task about guitar.',
    'Search the most recent notes.',
])
def test_latest_unrelated_items_do_not_route_to_vision(message):
    assert not should_use_latest_image(message)


@pytest.mark.asyncio
@pytest.mark.parametrize('tool_name', ['remember_memory', 'replace_memory'])
async def test_memory_tools_accept_legacy_fact_type(isolated_db, monkeypatch, tool_name):
    from database.long_term_memory import get_long_term_memory, save_long_term_memory
    from services import memory
    from tools.memory_tools import MEMORY_TOOLS, execute_memory_tool

    monkeypatch.setattr(memory, 'generate_passage_embeddings', lambda texts: [])
    args = {'content': 'I prefer blue notebook covers.', 'memory_type': 'fact', 'importance': 3}
    if tool_name == 'replace_memory':
        args['memory_id'] = await save_long_term_memory(101, 'I prefer amber notebook covers.')

    # The provider rejected this legacy type before the tool could execute.
    schema = next(tool['function']['parameters'] for tool in MEMORY_TOOLS
                  if tool['function']['name'] == tool_name)
    assert args['memory_type'] in schema['properties']['memory_type']['enum']
    result = json.loads(await execute_memory_tool(101, tool_name, args))
    assert result['success'] is True
    row = await get_long_term_memory(101, result['memory_id'])
    assert row[1:3] == ('I prefer blue notebook covers.', 'stable_fact')
    assert await get_long_term_memory(202, result['memory_id']) is None
    if tool_name == 'replace_memory':
        assert result['memory_id'] == args['memory_id']
