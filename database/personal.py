import psycopg
from psycopg.rows import dict_row
from config import DATABASE_URL


async def fetch(query, params=()):
    async with await psycopg.AsyncConnection.connect(DATABASE_URL, row_factory=dict_row, connect_timeout=10, options="-c statement_timeout=30000") as conn:
        cursor = await conn.execute(query, params)
        return await cursor.fetchall() if cursor.description else []


async def get_timezone(user_id):
    rows = await fetch('SELECT timezone FROM user_preferences WHERE telegram_user_id = %s', (user_id,))
    return rows[0]['timezone'] if rows else None


async def set_timezone(user_id, zone):
    from zoneinfo import ZoneInfo
    ZoneInfo(zone)
    await fetch('''INSERT INTO user_preferences VALUES (%s, %s)
        ON CONFLICT (telegram_user_id) DO UPDATE SET timezone = EXCLUDED.timezone''', (user_id, zone))


async def record_tool(user_id, tool, success, duration_ms):
    await fetch('''INSERT INTO tool_events (telegram_user_id, tool, success, duration_ms)
                   VALUES (%s, %s, %s, %s)''', (user_id, tool, success, duration_ms))
