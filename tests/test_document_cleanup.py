"""Single-document removal must preserve every other document and user."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot import commands
from database.documents import save_document


def telegram(args=(), user_id=101):
    return (SimpleNamespace(effective_user=SimpleNamespace(id=user_id),
                            message=SimpleNamespace(reply_text=AsyncMock())),
            SimpleNamespace(args=list(args)))


def replies(update):
    return '\n'.join(call.args[0] for call in update.message.reply_text.await_args_list)


@pytest.mark.asyncio
async def test_confirmed_single_delete_preserves_existing_documents_and_other_users(isolated_db):
    first = await save_document(101, 'existing.pdf', 'pdf', ['Preserve first'])
    second = await save_document(101, 'existing.pdf', 'pdf', ['Preserve second'])
    fixture = await save_document(101, 'smoke_test.txt', 'txt', ['Remove only this'])
    other = await save_document(202, 'private.txt', 'txt', ['Other user'])
    update, context = telegram()
    await commands.files_command(update, context)
    assert f'{first}. existing.pdf' in replies(update)
    assert f'{second}. existing.pdf' in replies(update)
    assert f'{fixture}. smoke_test.txt' in replies(update)
    assert 'private.txt' not in replies(update)

    update, context = telegram([str(fixture)])
    await commands.delete_file_command(update, context)
    assert f'/deletefile {fixture} confirm' in replies(update)
    assert 'smoke_test.txt' in replies(update)
    assert len(await isolated_db('SELECT id FROM documents')) == 4

    for args in ([str(other)], [str(other), 'confirm']):
        update, context = telegram(args)
        await commands.delete_file_command(update, context)
        assert replies(update) == 'File not found.'

    update, context = telegram([str(fixture), 'confirm'])
    await commands.delete_file_command(update, context)
    assert f'Deleted file {fixture}' in replies(update)
    assert {row['id'] for row in await isolated_db('SELECT id FROM documents')} == {first, second, other}
    assert {row['document_id'] for row in await isolated_db('SELECT document_id FROM document_chunks')} == {first, second, other}
    update, context = telegram([str(fixture), 'confirm'])
    await commands.delete_file_command(update, context)
    assert replies(update) == 'File not found.'


@pytest.mark.asyncio
@pytest.mark.parametrize('args', [[], ['all'], ['0'], ['-1'], [str(2**63)],
                                 ['1', 'yes'], ['1', 'confirm', 'extra']])
async def test_invalid_delete_command_never_accesses_database(monkeypatch, args):
    get = AsyncMock()
    delete = AsyncMock()
    monkeypatch.setattr(commands, 'get_document', get)
    monkeypatch.setattr(commands, 'delete_document', delete)
    update, context = telegram(args)
    await commands.delete_file_command(update, context)
    assert 'Usage:' in replies(update)
    get.assert_not_awaited()
    delete.assert_not_awaited()


def test_deletefile_handler_is_registered_without_starting_bot():
    from bot.app import build_application
    application = build_application()
    assert any('deletefile' in getattr(handler, 'commands', [])
               for handlers in application.handlers.values() for handler in handlers)
