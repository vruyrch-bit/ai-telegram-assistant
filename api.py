import os

import psycopg

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Query,
)

from fastapi.security import APIKeyHeader


# ==================================================
# ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")


if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL was not found in .env"
    )

if not ADMIN_API_KEY:
    raise ValueError(
        "ADMIN_API_KEY was not found in .env"
    )


# ==================================================
# FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="AI Telegram Bot API",
    description=(
        "REST API for the AI Telegram Bot, "
        "PostgreSQL memory, and administration."
    ),
    version="1.1.0",
)


# ==================================================
# API KEY AUTHENTICATION
# ==================================================

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


async def verify_api_key(
    api_key: str = Depends(api_key_header),
):

    if api_key != ADMIN_API_KEY:

        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )

    return api_key


# ==================================================
# ROOT
# ==================================================

@app.get("/")
async def root():

    return {
        "service": "AI Telegram Bot API",
        "status": "running",
        "version": "1.1.0",
    }


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/health")
async def health():

    try:

        async with await psycopg.AsyncConnection.connect(
            DATABASE_URL
        ) as connection:

            cursor = await connection.execute(
                "SELECT 1"
            )

            await cursor.fetchone()

        return {
            "status": "healthy",
            "database": "connected",
        }

    except Exception as error:

        print(
            f"Database health error: {error}"
        )

        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        )


# ==================================================
# PROTECTED STATISTICS
# ==================================================

@app.get("/stats")
async def stats(
    api_key: str = Depends(verify_api_key),
):

    try:

        async with await psycopg.AsyncConnection.connect(
            DATABASE_URL
        ) as connection:

            cursor = await connection.execute(
                """
                SELECT
                    COUNT(*) AS total_messages,
                    COUNT(
                        DISTINCT telegram_user_id
                    ) AS total_users,
                    COUNT(*) FILTER (
                        WHERE role = 'user'
                    ) AS user_messages,
                    COUNT(*) FILTER (
                        WHERE role IN (
                            'assistant',
                            'model'
                        )
                    ) AS ai_messages
                FROM messages
                """
            )

            row = await cursor.fetchone()

        return {
            "total_messages": row[0],
            "total_users": row[1],
            "user_messages": row[2],
            "ai_messages": row[3],
        }

    except Exception as error:

        print(
            f"Stats error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail="Could not retrieve statistics",
        )


# ==================================================
# LIST USERS
# ==================================================

@app.get("/users")
async def get_users(
    api_key: str = Depends(verify_api_key),
):

    try:

        async with await psycopg.AsyncConnection.connect(
            DATABASE_URL
        ) as connection:

            cursor = await connection.execute(
                """
                SELECT
                    telegram_user_id,
                    COUNT(*) AS message_count,
                    MAX(created_at) AS last_activity
                FROM messages
                GROUP BY telegram_user_id
                ORDER BY last_activity DESC
                """
            )

            rows = await cursor.fetchall()

        users = []

        for row in rows:

            users.append(
                {
                    "telegram_user_id": row[0],
                    "message_count": row[1],
                    "last_activity": row[2],
                }
            )

        return {
            "users": users
        }

    except Exception as error:

        print(
            f"Users error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail="Could not retrieve users",
        )


# ==================================================
# USER CONVERSATION
# ==================================================

@app.get(
    "/users/{telegram_user_id}/messages"
)
async def get_user_messages(
    telegram_user_id: int,

    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),

    api_key: str = Depends(
        verify_api_key
    ),
):

    try:

        async with await psycopg.AsyncConnection.connect(
            DATABASE_URL
        ) as connection:

            cursor = await connection.execute(
                """
                SELECT
                    id,
                    role,
                    content,
                    created_at
                FROM messages
                WHERE telegram_user_id = %s
                ORDER BY id DESC
                LIMIT %s
                """,
                (
                    telegram_user_id,
                    limit,
                ),
            )

            rows = await cursor.fetchall()

        if not rows:

            raise HTTPException(
                status_code=404,
                detail="User or messages not found",
            )

        rows.reverse()

        messages = []

        for row in rows:

            messages.append(
                {
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "created_at": row[3],
                }
            )

        return {
            "telegram_user_id": telegram_user_id,
            "message_count": len(messages),
            "messages": messages,
        }

    except HTTPException:
        raise

    except Exception as error:

        print(
            f"Conversation error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not retrieve "
                "conversation"
            ),
        )


# ==================================================
# DELETE USER MEMORY
# ==================================================

@app.delete(
    "/users/{telegram_user_id}/messages"
)
async def delete_user_messages(
    telegram_user_id: int,
    api_key: str = Depends(
        verify_api_key
    ),
):

    try:

        async with await psycopg.AsyncConnection.connect(
            DATABASE_URL
        ) as connection:

            cursor = await connection.execute(
                """
                DELETE FROM messages
                WHERE telegram_user_id = %s
                RETURNING id
                """,
                (
                    telegram_user_id,
                ),
            )

            deleted_rows = (
                await cursor.fetchall()
            )

        if not deleted_rows:

            raise HTTPException(
                status_code=404,
                detail="No messages found",
            )

        return {
            "status": "deleted",
            "telegram_user_id": telegram_user_id,
            "deleted_messages": len(
                deleted_rows
            ),
        }

    except HTTPException:
        raise

    except Exception as error:

        print(
            f"Delete error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail="Could not delete messages",
        )
