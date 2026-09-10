import asyncio
import json
import logging
from datetime import datetime, timezone

from database.long_term_memory import (
    list_long_term_memories,
    list_long_term_memories_with_embeddings,
    get_long_term_memory,
    save_long_term_memory,
    replace_long_term_memory,
    touch_long_term_memories,
)

from rag.embeddings import (
    generate_passage_embeddings,
    generate_query_embedding,
    cosine_similarity,
)


logger = logging.getLogger(__name__)


MEMORY_RETRIEVAL_LIMIT = 5
MEMORY_CANDIDATE_LIMIT = 100
MIN_MEMORY_SIMILARITY = 0.35
MEMORY_DUPLICATE_THRESHOLD = 0.93

# Relevance remains dominant; metadata only adds bounded bonuses.
MEMORY_RECENCY_BONUS = 0.05
MEMORY_ACCESS_BONUS = 0.03
MEMORY_RECENCY_HALF_LIFE_DAYS = 30.0
MEMORY_ACCESS_HALF_LIFE_DAYS = 7.0


def memory_time_decay(timestamp, now, half_life_days):
    """Missing timestamps add no bonus; treat legacy naive values as UTC."""
    if timestamp is None:
        return 0.0
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - timestamp).total_seconds() / 86400.0)
    return 2.0 ** (-age_days / half_life_days)



async def remember_user_memory(
    telegram_user_id: int,
    content: str,
    memory_type: str = "fact",
    importance: int = 3,
    source: str = "explicit_user",
):
    content = content.strip()

    if not content:
        raise ValueError(
            "Memory content cannot be empty."
        )

    embedding = None

    try:
        embeddings = await asyncio.to_thread(
            generate_passage_embeddings,
            [content],
        )

        if embeddings:
            embedding = embeddings[0]

    except Exception:
        logger.exception(
            "Memory embedding generation failed "
            "user_id=%s",
            telegram_user_id,
        )

    if embedding is not None:
        duplicate = (
            await find_semantic_duplicate(
                telegram_user_id,
                embedding,
            )
        )

        if duplicate:
            logger.info(
                "Semantic memory duplicate suppressed "
                "user_id=%s existing_memory_id=%s "
                "similarity=%.3f",
                telegram_user_id,
                duplicate["id"],
                duplicate["similarity"],
            )

            existing_type = (
                duplicate["memory_type"]
            )

            resolved_type = (
                memory_type
                if (
                    existing_type
                    in {
                        "fact",
                        "stable_fact",
                    }
                    and memory_type
                    not in {
                        "fact",
                        "stable_fact",
                    }
                )
                else existing_type
            )

            resolved_importance = max(
                int(
                    duplicate[
                        "importance"
                    ]
                ),
                int(importance),
            )

            resolved_source = (
                "explicit_user"
                if source == "explicit_user"
                else duplicate["source"]
            )

            await save_long_term_memory(
                telegram_user_id,
                duplicate["content"],
                memory_type=resolved_type,
                importance=resolved_importance,
                embedding=duplicate["embedding"],
                source=resolved_source,
            )

            return duplicate["id"]

    return await save_long_term_memory(
        telegram_user_id,
        content,
        memory_type=memory_type,
        importance=importance,
        embedding=embedding,
        source=source,
    )


async def replace_user_memory(
    telegram_user_id: int,
    memory_id: int,
    content: str,
    memory_type: str,
    importance: int = 3,
    source: str = "automatic",
):
    content = content.strip()

    if not content:
        raise ValueError(
            "Memory content cannot be empty."
        )

    embedding = None

    try:
        embeddings = await asyncio.to_thread(
            generate_passage_embeddings,
            [content],
        )

        if embeddings:
            embedding = embeddings[0]

    except Exception:
        logger.exception(
            "Replacement memory embedding failed "
            "user_id=%s memory_id=%s",
            telegram_user_id,
            memory_id,
        )

    return await replace_long_term_memory(
        telegram_user_id,
        memory_id,
        content,
        memory_type,
        importance,
        embedding=embedding,
        source=source,
    )

def parse_memory_embedding(
    raw_embedding,
):
    if not raw_embedding:
        return None

    if isinstance(
        raw_embedding,
        str,
    ):
        try:
            return json.loads(
                raw_embedding
            )

        except json.JSONDecodeError:
            return None

    return raw_embedding


async def find_semantic_duplicate(
    telegram_user_id: int,
    new_embedding,
):
    memories = (
        await list_long_term_memories_with_embeddings(
            telegram_user_id,
            limit=MEMORY_CANDIDATE_LIMIT,
        )
    )

    best_match = None

    for (
        memory_id,
        content,
        memory_type,
        importance,
        raw_embedding,
        embedding_model,
        source,
        updated_at,
    ) in memories:

        embedding = (
            parse_memory_embedding(
                raw_embedding
            )
        )

        if embedding is None:
            try:
                generated = (
                    await asyncio.to_thread(
                        generate_passage_embeddings,
                        [content],
                    )
                )

                if generated:
                    embedding = generated[0]

                    await save_long_term_memory(
                        telegram_user_id,
                        content,
                        memory_type=memory_type,
                        importance=importance,
                        embedding=embedding,
                        source=source,
                    )

            except Exception:
                logger.exception(
                    "Duplicate-check embedding "
                    "backfill failed "
                    "user_id=%s memory_id=%s",
                    telegram_user_id,
                    memory_id,
                )

                continue

        if embedding is None:
            continue

        similarity = cosine_similarity(
            new_embedding,
            embedding,
        )

        if (
            best_match is None
            or similarity
            > best_match["similarity"]
        ):
            best_match = {
                "id": memory_id,
                "content": content,
                "memory_type": memory_type,
                "importance": importance,
                "embedding": embedding,
                "source": source,
                "similarity": similarity,
            }

    if (
        best_match
        and best_match["similarity"]
        >= MEMORY_DUPLICATE_THRESHOLD
    ):
        return best_match

    return None


async def retrieve_relevant_memories(
    telegram_user_id: int,
    query: str,
    limit: int = MEMORY_RETRIEVAL_LIMIT,
):
    query = query.strip()

    if not query:
        return []

    memories = await list_long_term_memories(
        telegram_user_id,
        limit=MEMORY_CANDIDATE_LIMIT,
    )

    if not memories:
        return []

    try:
        query_embedding = (
            await asyncio.to_thread(
                generate_query_embedding,
                query,
            )
        )

    except Exception:
        logger.exception(
            "Memory query embedding failed "
            "user_id=%s",
            telegram_user_id,
        )
        return []

    scored = []
    now = datetime.now(timezone.utc)

    for (
        memory_id,
        content,
        memory_type,
        importance,
        source,
        created_at,
        updated_at,
    ) in memories:

        full_memory = (
            await get_long_term_memory(
                telegram_user_id,
                memory_id,
            )
        )

        if not full_memory:
            continue

        raw_embedding = (
            full_memory[4]
        )

        embedding = (
            parse_memory_embedding(
                raw_embedding
            )
        )

        # Existing memories created before semantic
        # retrieval may not have embeddings yet.
        if embedding is None:
            try:
                generated = (
                    await asyncio.to_thread(
                        generate_passage_embeddings,
                        [content],
                    )
                )

                if generated:
                    embedding = generated[0]

                    await save_long_term_memory(
                        telegram_user_id,
                        content,
                        memory_type=memory_type,
                        importance=importance,
                        embedding=embedding,
                        source=source,
                    )

            except Exception:
                logger.exception(
                    "Memory embedding backfill failed "
                    "user_id=%s memory_id=%s",
                    telegram_user_id,
                    memory_id,
                )
                continue

        if embedding is None:
            continue

        similarity = cosine_similarity(
            query_embedding,
            embedding,
        )

        importance_bonus = (
            max(
                1,
                min(
                    int(importance),
                    5,
                ),
            )
            / 5.0
        ) * 0.05

        recency_bonus = MEMORY_RECENCY_BONUS * memory_time_decay(
            full_memory[8] or full_memory[7],
            now,
            MEMORY_RECENCY_HALF_LIFE_DAYS,
        )
        access_bonus = MEMORY_ACCESS_BONUS * memory_time_decay(
            full_memory[9],
            now,
            MEMORY_ACCESS_HALF_LIFE_DAYS,
        )
        final_score = (
            similarity
            + importance_bonus
            + recency_bonus
            + access_bonus
        )

        if (
            similarity
            < MIN_MEMORY_SIMILARITY
        ):
            continue

        scored.append(
            {
                "id": memory_id,
                "content": content,
                "memory_type": memory_type,
                "importance": importance,
                "similarity": similarity,
                "score": final_score,
            }
        )

    scored.sort(
        key=lambda item:
            item["score"],
        reverse=True,
    )

    selected_memories = scored[:limit]

    if selected_memories:
        try:
            await touch_long_term_memories(
                telegram_user_id,
                [
                    memory["id"]
                    for memory in selected_memories
                ],
            )

        except Exception:
            logger.exception(
                "Failed to update memory access timestamps "
                "user_id=%s",
                telegram_user_id,
            )

    return selected_memories


async def build_memory_context(
    telegram_user_id: int,
    query: str,
):
    memories = (
        await retrieve_relevant_memories(
            telegram_user_id,
            query,
        )
    )

    if not memories:
        return None

    lines = []

    for memory in memories:
        lines.append(
            (
                f"• [memory_id={memory['id']}] "
                f"{memory['content']} "
                f"[type={memory['memory_type']}, "
                f"importance={memory['importance']}/5]"
            )
        )

    return "\n".join(
        lines
    )
