"""Live replacement failures involving existing or forgotten memory content."""
import asyncio

import psycopg
import pytest

from database.long_term_memory import (
    forget_long_term_memory,
    list_long_term_memories,
    replace_long_term_memory,
    save_long_term_memory,
)


@pytest.mark.asyncio
@pytest.mark.parametrize('forgotten', [False, True])
async def test_replace_reuses_duplicate_without_leaving_old_preference(isolated_db, forgotten):
    old = await save_long_term_memory(101, 'I prefer amber covers')
    target = await save_long_term_memory(101, 'I prefer blue covers')
    other = await save_long_term_memory(202, 'I prefer blue covers')
    if forgotten:
        await forget_long_term_memory(101, target)

    result = await replace_long_term_memory(101, old, 'I prefer blue covers', 'preference', 4)
    assert result == target
    rows = await list_long_term_memories(101)
    assert [(row[0], row[1], row[2], row[3]) for row in rows] == [
        (target, 'I prefer blue covers', 'preference', 4)]
    assert [row[0] for row in await list_long_term_memories(202)] == [other]
    assert len(await isolated_db('SELECT id FROM long_term_memories')) == 3

    # Changing back to the previous preference must work too.
    assert await replace_long_term_memory(101, target, 'I prefer amber covers', 'preference', 3) == old
    assert [row[0] for row in await list_long_term_memories(101)] == [old]


@pytest.mark.asyncio
async def test_invalid_source_cannot_revive_a_forgotten_duplicate(isolated_db):
    source = await save_long_term_memory(202, 'Amber covers')
    target = await save_long_term_memory(101, 'Blue covers')
    await forget_long_term_memory(101, target)
    assert await replace_long_term_memory(101, source, 'Blue covers', 'fact', 3) is None
    assert await list_long_term_memories(101) == []
    assert [row[0] for row in await list_long_term_memories(202)] == [source]


@pytest.mark.asyncio
async def test_failed_duplicate_update_rolls_back_source_deactivation(isolated_db):
    source = await save_long_term_memory(101, 'Amber covers')
    target = await save_long_term_memory(101, 'Blue covers')
    await forget_long_term_memory(101, target)
    with pytest.raises(psycopg.errors.NotNullViolation):
        await replace_long_term_memory(101, source, 'Blue covers', None, 3)
    assert [row[0] for row in await list_long_term_memories(101)] == [source]


@pytest.mark.asyncio
async def test_concurrent_save_and_replace_share_one_active_target(isolated_db):
    source = await save_long_term_memory(101, 'Amber covers')
    saved, replaced = await asyncio.gather(
        save_long_term_memory(101, 'Blue covers'),
        replace_long_term_memory(101, source, 'Blue covers', 'preference', 3),
    )
    assert saved == replaced
    assert [row[0] for row in await list_long_term_memories(101)] == [replaced]
