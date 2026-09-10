"""Durable reminder delivery. A transaction lock prevents concurrent claims.

Delivery is at-least-once: a crash after Telegram accepts a message but before
commit can duplicate it. Telegram sendMessage has no idempotency key.
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone

import psycopg
from psycopg.rows import dict_row
from telegram.error import Forbidden, BadRequest, RetryAfter
from config import DATABASE_URL
from services.scheduling import next_occurrence

logger = logging.getLogger(__name__)


async def deliver_one(bot):
    async with await psycopg.AsyncConnection.connect(DATABASE_URL, row_factory=dict_row, connect_timeout=10, options="-c statement_timeout=30000") as conn:
        await conn.execute("SET LOCAL lock_timeout = '2s'")
        cursor = await conn.execute('''SELECT * FROM reminders
            WHERE status = 'pending' AND due_at <= NOW()
            AND (retry_at IS NULL OR retry_at <= NOW())
            ORDER BY due_at, id LIMIT 1 FOR UPDATE SKIP LOCKED''')
        job = await cursor.fetchone()
        if not job:
            return False
        try:
            await asyncio.wait_for(bot.send_message(
                chat_id=job['telegram_user_id'],
                text=f"⏰ Reminder #{job['id']}\n{job['content']}",
            ), timeout=20)
        except Exception as error:
            attempts = job['attempts'] + 1
            terminal = isinstance(error, (Forbidden, BadRequest)) or attempts >= 5
            delay = min(3600, 30 * 2 ** attempts)
            if isinstance(error, RetryAfter):
                value = error.retry_after
                delay = value.total_seconds() if isinstance(value, timedelta) else value
            await conn.execute('''UPDATE reminders SET attempts = %s, status = %s,
                retry_at = %s, last_error = %s WHERE id = %s''',
                (attempts, 'failed' if terminal else 'pending',
                 datetime.now(timezone.utc) + timedelta(seconds=delay),
                 type(error).__name__, job['id']))
            logger.warning('Reminder delivery failed id=%s error=%s', job['id'], type(error).__name__)
        else:
            due = next_occurrence(job['due_at'], job['timezone'], job['recurrence'])
            await conn.execute('''UPDATE reminders SET status = %s, due_at = %s,
                attempts = 0, retry_at = NULL, last_error = NULL WHERE id = %s''',
                ('pending' if due else 'sent', due or job['due_at'], job['id']))
        return True


async def run_jobs(bot):
    while True:
        try:
            for _ in range(20):
                if not await deliver_one(bot):
                    break
        except Exception as error:
            logger.warning('Reminder worker retrying after %s', type(error).__name__)
        await asyncio.sleep(5)
