import os

import numpy as np
import pytest


# ==================================================
# FAKE ENVIRONMENT VARIABLES
# ==================================================
# main.py checks these during import.
# These are test-only fake values.

os.environ.setdefault(
    "TELEGRAM_TOKEN",
    "test-telegram-token",
)

os.environ.setdefault(
    "GROQ_API_KEY",
    "test-groq-key",
)

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://test:test@localhost:5432/test",
)


import main
from tools import task_tools as task_tools_module


# ==================================================
# TELEGRAM FORMATTING
# ==================================================

def test_clean_telegram_text_removes_markdown():

    text = (
        "# Heading\n"
        "**Bold text**\n"
        "- First item\n"
        "- Second item\n"
        "`code`"
    )

    result = main.clean_telegram_text(
        text
    )

    assert "#" not in result
    assert "**" not in result
    assert "`" not in result

    assert "Heading" in result
    assert "Bold text" in result
    assert "• First item" in result
    assert "• Second item" in result


def test_clean_telegram_text_removes_extra_blank_lines():

    text = (
        "First\n\n\n\n\nSecond"
    )

    result = main.clean_telegram_text(
        text
    )

    assert "\n\n\n" not in result
    assert "First" in result
    assert "Second" in result


# ==================================================
# DOCUMENT CHUNKING
# ==================================================

def test_chunk_text_short_document():

    text = (
        "This is a short document."
    )

    chunks = main.chunk_text(
        text
    )

    assert len(chunks) == 1

    assert chunks[0] == (
        "This is a short document."
    )


def test_chunk_text_long_document():

    text = (
        "Artificial intelligence systems "
        "can process information. "
        * 100
    )

    chunks = main.chunk_text(
        text
    )

    assert len(chunks) > 1

    for chunk in chunks:

        assert chunk.strip()

        assert len(chunk) <= (
            main.DOCUMENT_CHUNK_SIZE
            + 10
        )


def test_chunk_text_empty_string():

    chunks = main.chunk_text("")

    assert chunks == []


# ==================================================
# QUERY WORD EXTRACTION
# ==================================================

def test_question_words_removes_stop_words():

    result = main.question_words(
        "What are the important Python "
        "and PostgreSQL skills?"
    )

    assert "python" in result
    assert "postgresql" in result
    assert "skills" in result

    assert "what" not in result
    assert "the" not in result
    assert "and" not in result


# ==================================================
# COSINE SIMILARITY
# ==================================================

def test_cosine_similarity_identical_vectors():

    vector = [
        1.0
    ] * main.EMBEDDING_DIMENSIONS

    result = main.cosine_similarity(
        vector,
        vector,
    )

    assert result == pytest.approx(
        1.0,
        abs=0.0001,
    )


def test_cosine_similarity_opposite_vectors():

    vector_a = [
        1.0
    ] * main.EMBEDDING_DIMENSIONS

    vector_b = [
        -1.0
    ] * main.EMBEDDING_DIMENSIONS

    result = main.cosine_similarity(
        vector_a,
        vector_b,
    )

    assert result == pytest.approx(
        -1.0,
        abs=0.0001,
    )


def test_cosine_similarity_wrong_dimensions():

    result = main.cosine_similarity(
        [1.0, 2.0],
        [1.0, 2.0],
    )

    assert result == 0.0


# ==================================================
# KEYWORD RANKING
# ==================================================

def test_keyword_score_matching_content():

    query_words = {
        "python",
        "postgresql",
    }

    result = (
        main.calculate_keyword_score(
            (
                "The backend uses Python "
                "with PostgreSQL."
            ),
            query_words,
        )
    )

    assert result == 1.0


def test_keyword_score_partial_match():

    query_words = {
        "python",
        "kubernetes",
    }

    result = (
        main.calculate_keyword_score(
            (
                "The application "
                "is written in Python."
            ),
            query_words,
        )
    )

    assert result == 0.5


def test_keyword_score_no_match():

    query_words = {
        "python",
        "postgresql",
    }

    result = (
        main.calculate_keyword_score(
            "The restaurant serves pizza.",
            query_words,
        )
    )

    assert result == 0.0


# ==================================================
# IMAGE FOLLOW-UP DETECTION
# ==================================================

def test_detect_image_followup():

    assert (
        main.should_use_latest_image(
            "What is happening in this image?"
        )
        is True
    )


def test_detect_photo_followup():

    assert (
        main.should_use_latest_image(
            "Can you explain the photo?"
        )
        is True
    )


def test_normal_message_is_not_image_followup():

    assert (
        main.should_use_latest_image(
            "Explain how PostgreSQL works."
        )
        is False
    )


# ==================================================
# IMAGE NORMALIZATION
# ==================================================

def test_image_normalization():

    from PIL import Image
    import io

    image = Image.new(
        "RGB",
        (
            3000,
            2000,
        ),
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    (
        normalized_bytes,
        normalized_image,
    ) = main.normalize_image_for_vision(
        buffer.getvalue()
    )

    assert isinstance(
        normalized_bytes,
        bytes,
    )

    assert (
        normalized_image.width
        <= main.MAX_IMAGE_DIMENSION
    )

    assert (
        normalized_image.height
        <= main.MAX_IMAGE_DIMENSION
    )

    assert (
        normalized_image.mode
        == "RGB"
    )


# ==================================================
# TASK TOOL CALLING
# ==================================================

@pytest.mark.asyncio
async def test_create_task_tool(
    monkeypatch,
):

    async def fake_create_task(
        telegram_user_id,
        title,
        due_date=None,
    ):

        return 42


    monkeypatch.setattr(
        task_tools_module,
        "create_task",
        fake_create_task,
    )


    result = (
        await main.execute_task_tool(
            telegram_user_id=123,

            tool_name="create_task",

            arguments={
                "title":
                    "Finish calculus homework",

                "due_date":
                    "Friday",
            },
        )
    )


    assert '"success": true' in result
    assert '"task_id": 42' in result
    assert (
        "Finish calculus homework"
        in result
    )


@pytest.mark.asyncio
async def test_list_tasks_tool(
    monkeypatch,
):

    async def fake_get_tasks(
        telegram_user_id,
    ):

        return [
            (
                1,
                "Finish calculus",
                "open",
                "Friday",
            ),

            (
                2,
                "Chemistry report",
                "done",
                None,
            ),
        ]


    monkeypatch.setattr(
        task_tools_module,
        "get_tasks",
        fake_get_tasks,
    )


    result = (
        await main.execute_task_tool(
            telegram_user_id=123,

            tool_name="list_tasks",

            arguments={},
        )
    )


    assert '"success": true' in result

    assert "Finish calculus" in result

    assert "Chemistry report" in result


@pytest.mark.asyncio
async def test_complete_task_tool(
    monkeypatch,
):

    async def fake_complete_task(
        telegram_user_id,
        task_id,
    ):

        return (
            "Finish calculus homework"
        )


    monkeypatch.setattr(
        task_tools_module,
        "complete_task",
        fake_complete_task,
    )


    result = (
        await main.execute_task_tool(
            telegram_user_id=123,

            tool_name="complete_task",

            arguments={
                "task_id": 5
            },
        )
    )


    assert '"success": true' in result

    assert '"status": "done"' in result


@pytest.mark.asyncio
async def test_delete_task_tool(
    monkeypatch,
):

    async def fake_delete_task(
        telegram_user_id,
        task_id,
    ):

        return "Old task"


    monkeypatch.setattr(
        task_tools_module,
        "delete_task",
        fake_delete_task,
    )


    result = (
        await main.execute_task_tool(
            telegram_user_id=123,

            tool_name="delete_task",

            arguments={
                "task_id": 9
            },
        )
    )


    assert '"success": true' in result

    assert '"deleted": true' in result


@pytest.mark.asyncio
async def test_unknown_tool():

    result = (
        await main.execute_task_tool(
            telegram_user_id=123,

            tool_name="not_a_real_tool",

            arguments={},
        )
    )

    assert '"success": false' in result
    assert "Unknown tool" in result
