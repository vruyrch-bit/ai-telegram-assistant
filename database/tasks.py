import psycopg

from config import DATABASE_URL


async def create_task(
    telegram_user_id: int,
    title: str,
    due_date=None,
    *,
    priority: int = 3,
    project: str = '',
    notes: str = '',
    recurrence: str = 'none',
):
    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:
        cursor = await connection.execute(
            """
            INSERT INTO tasks (
                telegram_user_id,
                title,
                due_date,
                priority,
                project,
                notes,
                recurrence
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                telegram_user_id,
                title,
                due_date,
                priority,
                project,
                notes,
                recurrence,
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
                (status = 'open') DESC,
                priority DESC,
                id DESC
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
            RETURNING title, due_date, priority, project, notes, recurrence
            """,
            (
                task_id,
                telegram_user_id,
            ),
        )

        row = await cursor.fetchone()
        if row and row[5] != 'none':
            from datetime import date, timedelta
            previous = date.fromisoformat(row[1])
            step = 1 if row[5] == 'daily' else 7
            today = date.today()
            days = max(step, ((today - previous).days // step + 1) * step)
            next_due = previous + timedelta(days=days)
            await connection.execute("""INSERT INTO tasks
                (telegram_user_id, title, due_date, priority, project, notes, recurrence)
                VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (telegram_user_id, row[0], next_due.isoformat(), *row[2:]))

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


async def get_task_number_map(
    telegram_user_id: int,
):
    """Map private database IDs to user-visible task numbers."""
    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:
        cursor = await connection.execute(
            """
            SELECT id
            FROM tasks
            WHERE telegram_user_id = %s
            ORDER BY
                (status = 'open') DESC,
                priority DESC,
                id DESC
            LIMIT 100
            """,
            (telegram_user_id,),
        )
        rows = await cursor.fetchall()

    return {
        row[0]: number
        for number, row in enumerate(
            rows,
            start=1,
        )
    }


async def resolve_task_number(
    telegram_user_id: int,
    task_number: int,
):
    """Resolve visible task number to private database ID."""
    if task_number < 1:
        return None

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:
        cursor = await connection.execute(
            """
            SELECT id
            FROM tasks
            WHERE telegram_user_id = %s
            ORDER BY
                (status = 'open') DESC,
                priority DESC,
                id DESC
            OFFSET %s
            LIMIT 1
            """,
            (
                telegram_user_id,
                task_number - 1,
            ),
        )
        row = await cursor.fetchone()

    return row[0] if row else None
