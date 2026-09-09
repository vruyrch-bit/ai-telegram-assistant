import asyncio
import json
import logging
import re

import psycopg

from config import (
    DATABASE_URL,
    DOCUMENT_CHUNK_SIZE,
    DOCUMENT_CHUNK_OVERLAP,
    MAX_DOCUMENT_CONTEXT,
    DOCUMENT_RETRIEVAL_LIMIT,
    SUMMARY_CHUNK_LIMIT,
    SEMANTIC_WEIGHT,
    KEYWORD_WEIGHT,
    MIN_SEMANTIC_SCORE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSIONS,
)

from rag.embeddings import (
    generate_passage_embeddings,
    generate_query_embedding,
    cosine_similarity,
)


logger = logging.getLogger(__name__)


# ==================================================
# DOCUMENT CHUNKING
# ==================================================

def chunk_text(
    text: str,
):
    text = re.sub(
        r"\r\n?",
        "\n",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    text_length = len(
        text
    )

    while start < text_length:

        desired_end = min(
            start
            + DOCUMENT_CHUNK_SIZE,
            text_length,
        )

        end = desired_end

        if desired_end < text_length:

            minimum_break = (
                start
                + DOCUMENT_CHUNK_SIZE // 2
            )

            paragraph_break = (
                text.rfind(
                    "\n\n",
                    minimum_break,
                    desired_end,
                )
            )

            sentence_break = (
                text.rfind(
                    ". ",
                    minimum_break,
                    desired_end,
                )
            )

            space_break = (
                text.rfind(
                    " ",
                    minimum_break,
                    desired_end,
                )
            )

            if paragraph_break != -1:

                end = (
                    paragraph_break
                    + 2
                )

            elif sentence_break != -1:

                end = (
                    sentence_break
                    + 1
                )

            elif space_break != -1:

                end = space_break

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        if end >= text_length:
            break

        next_start = max(
            0,
            end
            - DOCUMENT_CHUNK_OVERLAP,
        )

        if next_start <= start:

            next_start = end

        start = next_start

    return chunks


# ==================================================
# KEYWORD SEARCH
# ==================================================

STOP_WORDS = {
    "the",
    "and",
    "that",
    "this",
    "what",
    "where",
    "when",
    "with",
    "from",
    "have",
    "does",
    "about",
    "into",
    "your",
    "would",
    "could",
    "should",
    "there",
    "they",
    "them",
    "then",
    "than",
    "are",
    "was",
    "were",
    "for",
    "how",
    "who",
    "why",
    "can",
    "tell",
    "please",
}


def question_words(
    question: str,
):
    words = re.findall(
        r"[A-Za-z0-9]+",
        question.lower(),
    )

    return {
        word

        for word in words

        if (
            len(word) >= 3
            and
            word not in STOP_WORDS
        )
    }


def calculate_keyword_score(
    content: str,
    query_words,
):
    if not query_words:
        return 0.0

    lower_content = (
        content.lower()
    )

    hits = 0

    for word in query_words:

        if re.search(
            rf"\b{re.escape(word)}\b",
            lower_content,
        ):

            hits += 1

    return (
        hits
        / len(query_words)
    )


# ==================================================
# EMBEDDING PARSING
# ==================================================

def parse_embedding(
    embedding_value,
):
    if not embedding_value:
        return None

    try:

        if isinstance(
            embedding_value,
            list,
        ):

            vector = (
                embedding_value
            )

        else:

            vector = json.loads(
                embedding_value
            )

        if (
            not isinstance(
                vector,
                list,
            )
            or
            len(vector)
            != EMBEDDING_DIMENSIONS
        ):

            return None

        return vector

    except Exception:

        return None


# ==================================================
# BACKFILL OLD EMBEDDINGS
# ==================================================

async def backfill_missing_embeddings(
    rows,
):
    missing = []

    for row in rows:

        (
            chunk_id,
            document_id,
            filename,
            chunk_index,
            content,
            embedding_value,
            embedding_model,
        ) = row

        current_embedding = (
            parse_embedding(
                embedding_value
            )
        )

        if (
            current_embedding is None
            or
            embedding_model
            != EMBEDDING_MODEL_NAME
        ):

            missing.append(
                (
                    chunk_id,
                    content,
                )
            )

    if not missing:
        return {}

    logger.info(
        "Backfilling embeddings chunks=%s",
        len(missing),
    )

    texts = [
        content

        for (
            chunk_id,
            content,
        ) in missing
    ]

    try:

        embeddings = (
            await asyncio.to_thread(
                generate_passage_embeddings,
                texts,
            )
        )

    except Exception:

        logger.exception(
            "Embedding backfill failed"
        )

        return {}

    new_embeddings = {}

    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        for (
            chunk_data,
            embedding,
        ) in zip(
            missing,
            embeddings,
        ):

            chunk_id = (
                chunk_data[0]
            )

            new_embeddings[
                chunk_id
            ] = embedding

            await connection.execute(
                """
                UPDATE document_chunks

                SET
                    embedding = %s,
                    embedding_model = %s

                WHERE id = %s
                """,
                (
                    json.dumps(
                        embedding
                    ),
                    EMBEDDING_MODEL_NAME,
                    chunk_id,
                ),
            )

    logger.info(
        "Embedding backfill complete chunks=%s",
        len(new_embeddings),
    )

    return new_embeddings


# ==================================================
# SUMMARY RETRIEVAL
# ==================================================

def select_summary_chunks(
    rows,
):
    if not rows:
        return []

    newest_document_id = (
        rows[0][1]
    )

    document_rows = [
        row

        for row in rows

        if row[1]
        == newest_document_id
    ]

    if (
        len(document_rows)
        <= SUMMARY_CHUNK_LIMIT
    ):

        return document_rows

    selected = []

    number_of_rows = len(
        document_rows
    )

    for i in range(
        SUMMARY_CHUNK_LIMIT
    ):

        index = round(
            i
            * (number_of_rows - 1)
            / (SUMMARY_CHUNK_LIMIT - 1)
        )

        row = document_rows[
            index
        ]

        if row not in selected:

            selected.append(
                row
            )

    return selected


# ==================================================
# DOCUMENT CONTEXT
# ==================================================

def build_document_context(
    selected_rows,
):
    context_parts = []

    total_length = 0

    for row in selected_rows:

        (
            chunk_id,
            document_id,
            filename,
            chunk_index,
            content,
            embedding_value,
            embedding_model,
        ) = row

        section = (
            f"[Source: {filename}, "
            f"chunk {chunk_index + 1}]\n"
            f"{content}"
        )

        if (
            total_length
            + len(section)
            > MAX_DOCUMENT_CONTEXT
        ):

            break

        context_parts.append(
            section
        )

        total_length += len(
            section
        )

    if not context_parts:
        return None

    return "\n\n".join(
        context_parts
    )


# ==================================================
# HYBRID SEMANTIC RAG
# ==================================================

async def get_document_context(
    telegram_user_id: int,
    question: str,
):
    async with await psycopg.AsyncConnection.connect(
        DATABASE_URL
    ) as connection:

        cursor = await connection.execute(
            """
            SELECT
                dc.id,
                d.id,
                d.filename,
                dc.chunk_index,
                dc.content,
                dc.embedding,
                dc.embedding_model

            FROM document_chunks dc

            JOIN documents d
                ON d.id = dc.document_id

            WHERE
                d.telegram_user_id = %s

            ORDER BY
                d.id DESC,
                dc.chunk_index ASC

            LIMIT 500
            """,
            (
                telegram_user_id,
            ),
        )

        rows = await cursor.fetchall()

    if not rows:
        return None

    lower_question = (
        question.lower()
    )

    summary_request = any(
        phrase in lower_question

        for phrase in (
            "summarize",
            "summarise",
            "summary",
            "overview of the document",
            "overview of this document",
            "main points",
            "key points",
            "summarize the pdf",
            "summarize this pdf",
            "summarize the file",
        )
    )

    if summary_request:

        selected_rows = (
            select_summary_chunks(
                rows
            )
        )

        logger.info(
            "Summary retrieval "
            "user_id=%s chunks=%s",
            telegram_user_id,
            len(selected_rows),
        )

        return build_document_context(
            selected_rows
        )

    query_words = (
        question_words(
            question
        )
    )

    backfilled_embeddings = (
        await backfill_missing_embeddings(
            rows
        )
    )

    query_embedding = None

    try:

        query_embedding = (
            await asyncio.to_thread(
                generate_query_embedding,
                question,
            )
        )

    except Exception:

        logger.exception(
            "Query embedding failed"
        )

    scored = []

    for row in rows:

        (
            chunk_id,
            document_id,
            filename,
            chunk_index,
            content,
            embedding_value,
            embedding_model,
        ) = row

        keyword_score = (
            calculate_keyword_score(
                content,
                query_words,
            )
        )

        semantic_score = 0.0

        if query_embedding is not None:

            chunk_embedding = (
                backfilled_embeddings.get(
                    chunk_id
                )
            )

            if chunk_embedding is None:

                if (
                    embedding_model
                    == EMBEDDING_MODEL_NAME
                ):

                    chunk_embedding = (
                        parse_embedding(
                            embedding_value
                        )
                    )

            if chunk_embedding is not None:

                semantic_score = (
                    cosine_similarity(
                        query_embedding,
                        chunk_embedding,
                    )
                )

        hybrid_score = (
            SEMANTIC_WEIGHT
            * semantic_score
            +
            KEYWORD_WEIGHT
            * keyword_score
        )

        scored.append(
            (
                hybrid_score,
                semantic_score,
                keyword_score,
                row,
            )
        )

    scored.sort(
        key=lambda item:
            item[0],
        reverse=True,
    )

    if not scored:
        return None

    best_semantic_score = (
        scored[0][1]
    )

    best_keyword_score = max(
        item[2]

        for item in scored
    )

    if (
        best_semantic_score
        < MIN_SEMANTIC_SCORE

        and

        best_keyword_score <= 0
    ):

        return None

    selected_rows = [
        item[3]

        for item in scored[
            :DOCUMENT_RETRIEVAL_LIMIT
        ]
    ]

    logger.info(
        "Hybrid RAG retrieval "
        "user_id=%s "
        "best_semantic=%.3f "
        "best_keyword=%.3f "
        "chunks=%s",
        telegram_user_id,
        best_semantic_score,
        best_keyword_score,
        len(selected_rows),
    )

    return build_document_context(
        selected_rows
    )
