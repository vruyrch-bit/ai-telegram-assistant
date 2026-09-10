"""Tests never inherit application credentials from the developer's .env."""
import importlib
import os

import pytest
import pytest_asyncio
from psycopg.conninfo import conninfo_to_dict

os.environ.update(
    PYTHON_DOTENV_DISABLED='1',
    TELEGRAM_TOKEN='123456:test-telegram-token',
    GROQ_API_KEY='test-groq-key',
    ADMIN_API_KEY='test-admin-key',
    DATABASE_URL='postgresql://test:test@127.0.0.1:1/test',
    AI_PROVIDER='groq',
    ALLOWED_USER_IDS='',
)


@pytest_asyncio.fixture
async def isolated_db(monkeypatch):
    url = os.getenv('BOT_TEST_DATABASE_URL')
    if not url:
        pytest.skip('Requires disposable BOT_TEST_DATABASE_URL')
    parts = conninfo_to_dict(url)
    host = parts.get('host', '')
    if not (host in {'127.0.0.1', 'localhost'} or host.startswith('/tmp/bot-validation-')):
        pytest.fail('Integration tests require a disposable local PostgreSQL instance')
    if not parts.get('dbname', '').startswith('bot_test'):
        pytest.fail('Use a disposable database whose name starts with bot_test')
    modules = [importlib.import_module(name) for name in (
        'config', 'api', 'database.core', 'database.upgrades', 'database.personal',
        'database.tasks', 'database.images', 'database.long_term_memory',
        'database.documents', 'database.memory', 'services.jobs', 'rag.retrieval',
    )]
    for module in modules:
        monkeypatch.setattr(module, 'DATABASE_URL', url)
    from database.core import initialize_database
    from database.long_term_memory import initialize_long_term_memory
    from database.upgrades import initialize_upgrades
    from database.personal import fetch
    await initialize_database()
    await initialize_long_term_memory()
    await initialize_upgrades()
    # This fixture is restricted to an explicitly supplied disposable test DB.
    await fetch('TRUNCATE messages, documents, tasks, latest_images, image_history, '
                'long_term_memories, reminders, knowledge_notes, user_preferences, '
                'tool_events RESTART IDENTITY CASCADE')
    return fetch
