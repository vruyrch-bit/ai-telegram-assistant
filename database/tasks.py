import psycopg

from config import DATABASE_URL


async def create_task(
    telegram_user_id: int,
    title: str,
    due_date=None,
):
    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:
        cursor = await connection.execute(
            """
            INSERT INTO tasks (
                telegram_user_id,
                title,
                due_date
            )
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (
                telegram_user_id,
                title,
                due_date,
            ),
        )

        row = await cursor.fetchone()

    return row[0]


async def get_tasks(
    telegram_user_id: int,
):
    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:
        cursor = await connection.execute(
            """
            SELECT
                id,
                title,
                status,
                due_date
            FROM tasks
            WHERE telegram_user_id = %s
            ORDER BY
                CASE
                    WHEN status = 'open'
                    THEN 0
                    ELSE 1
                END,
                id ASC
            """,
            (
                telegram_user_id,
            ),
        )

        return await cursor.fetchall()


async def complete_task(
    telegram_user_id: int,
    task_id: int,
):
    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:
        cursor = await connection.execute(
            """
            UPDATE tasks
            SET
                status = 'done',
                completed_at = CURRENT_TIMESTAMP
            WHERE
                id = %s
                AND telegram_user_id = %s
                AND status = 'open'
            RETURNING title
            """,
            (
                task_id,
                telegram_user_id,
            ),
        )

        row = await cursor.fetchone()

    if not row:
        return None

    return row[0]


async def delete_task(
    telegram_user_id: int,
    task_id: int,
):
    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:
        cursor = await connection.execute(
            """
            DELETE FROM tasks
            WHERE
                id = %s
                AND telegram_user_id = %s
            RETURNING title
            """,
            (
                task_id,
                telegram_user_id,
            ),
        )

        row = await cursor.fetchone()

    if not row:
        return None

    return row[0]
