import logging
import threading

import numpy as np

from fastembed import TextEmbedding

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSIONS,
    FASTEMBED_CACHE_DIR,
)


logger = logging.getLogger(__name__)


_embedding_model = None

_embedding_init_lock = threading.Lock()

_embedding_inference_lock = threading.Lock()


def get_embedding_model():
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    with _embedding_init_lock:
        if _embedding_model is not None:
            return _embedding_model

        logger.info(
            "Loading embedding model=%s",
            EMBEDDING_MODEL_NAME,
        )

        options = {
            "model_name":
                EMBEDDING_MODEL_NAME
        }

        if FASTEMBED_CACHE_DIR:
            options[
                "cache_dir"
            ] = FASTEMBED_CACHE_DIR

        _embedding_model = TextEmbedding(
            **options
        )

        logger.info(
            "Embedding model ready"
        )

    return _embedding_model


def generate_passage_embeddings(
    texts,
):
    if not texts:
        return []

    model = get_embedding_model()

    with _embedding_inference_lock:
        vectors = list(
            model.passage_embed(
                texts
            )
        )

    return [
        vector.astype(
            np.float32
        ).tolist()

        for vector in vectors
    ]


def generate_query_embedding(
    text: str,
):
    model = get_embedding_model()

    with _embedding_inference_lock:
        vectors = list(
            model.query_embed(
                [text]
            )
        )

    if not vectors:
        return None

    return (
        vectors[0]
        .astype(np.float32)
        .tolist()
    )


def cosine_similarity(
    vector_a,
    vector_b,
):
    try:
        a = np.asarray(
            vector_a,
            dtype=np.float32,
        )

        b = np.asarray(
            vector_b,
            dtype=np.float32,
        )

        if (
            len(a)
            != EMBEDDING_DIMENSIONS
            or
            len(b)
            != EMBEDDING_DIMENSIONS
        ):
            return 0.0

        denominator = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        if denominator == 0:
            return 0.0

        return float(
            np.dot(a, b)
            / denominator
        )

    except Exception:
        return 0.0
