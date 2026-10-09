"""
Zero-shot dense retrieval baseline over the ruling corpus, using a pretrained
multilingual embedding model (no fine-tuning).

Model: intfloat/multilingual-e5-small (118M params, 384-dim, CPU-feasible).
Requires "query: "/"passage: " prefixes on its inputs - that's this model's
own convention for asymmetric retrieval, not something we chose.

The model's max_seq_length is 512 tokens, but ruling `considerations` average
~9.7K characters and queries are longer still, so a single
embedding would only see the first ~18% of either. Both sides are therefore
chunked into fixed 512-token windows (see `lib.chunking`): every ruling chunk is
embedded and a ruling scores as its best-matching chunk (max), while a query's
chunk embeddings are mean-pooled into one vector.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib.atomic_dir import atomic_dir_write
from lib.decision_ids import read_decision_ids, write_decision_ids
from lib.dense.embedding_model import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    chunk_for_model,
    model,
)
from lib.index_cache import cache_name
from lib.ruling_chunks import load_ruling_chunks
from paths import INDEX_DIR

_CACHE_BASE = "dense_index"
_BATCH_SIZE = 32


@dataclass(frozen=True)
class DenseIndex:
    """
    Embeddings of every chunk of every ruling.

    :ivar chunk_embeddings: One L2-normalized row per chunk, grouped by ruling in
        `decision_ids` order.
    :ivar chunk_starts: The row in `chunk_embeddings` where each ruling's chunks begin,
        aligned with `decision_ids`; a ruling's chunks run until the next ruling's
        start.
    :ivar decision_ids: The ruling for each group of chunks.
    """

    chunk_embeddings: np.ndarray
    chunk_starts: np.ndarray
    decision_ids: list[str]


def build_dense_index(
    *, ruling_ids: frozenset[str] | None = None, model_path: Path | None = None
) -> DenseIndex:
    """
    Embed every 512-token chunk of the whole ruling corpus's `considerations` text.

    L2-normalized, so retrieval can use a plain dot product as cosine
    similarity. Downloads/expands the corpus first if necessary (see
    `lib.rulings`). Cached to disk under data/indexes/ (one directory per distinct
    set of rulings, see `lib.index_cache`) - idempotent,
    only built once (embedding the full chunked corpus takes roughly 6x as long as
    one embedding per ruling did: ~40 minutes on a GTX 1060).

    :param ruling_ids: Only index these rulings (a reduced corpus, for fast
        iteration); all of them if omitted.
    :param model_path: A fine-tuned model's directory to embed with; the pretrained
        model if omitted. The index is cached per model directory name, so a model
        must never be retrained under a name that already has a cached index.
    :returns: The index, with `decision_ids` aligned to the chunk groups.
    """
    base = _CACHE_BASE if model_path is None else f"{_CACHE_BASE}_{model_path.name}"
    embeddings_dir = INDEX_DIR / cache_name(base=base, ruling_ids=ruling_ids)
    if embeddings_dir.is_dir():
        return DenseIndex(
            chunk_embeddings=np.load(embeddings_dir / "chunk_embeddings.npy"),
            chunk_starts=np.load(embeddings_dir / "chunk_starts.npy"),
            decision_ids=read_decision_ids(directory=embeddings_dir),
        )

    chunks = load_ruling_chunks(ruling_ids=ruling_ids)
    chunk_embeddings = model(path=model_path).encode(
        [f"{PASSAGE_PREFIX}{text}" for text in chunks.texts],
        batch_size=_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    with atomic_dir_write(final_dir=embeddings_dir) as tmp_dir:
        tmp_dir.mkdir(parents=True)
        np.save(tmp_dir / "chunk_embeddings.npy", chunk_embeddings)
        np.save(tmp_dir / "chunk_starts.npy", chunks.chunk_starts)
        write_decision_ids(directory=tmp_dir, decision_ids=chunks.decision_ids)

    return DenseIndex(
        chunk_embeddings=chunk_embeddings,
        chunk_starts=chunks.chunk_starts,
        decision_ids=chunks.decision_ids,
    )


def _embed_query(*, query: str, model_path: Path | None) -> np.ndarray:
    """
    Embed every 512-token chunk of `query` and mean-pool them into one vector.

    :param query: The query text.
    :param model_path: The fine-tuned model's directory; the pretrained model if None.
    :returns: A single L2-normalized embedding for the whole query.
    """
    chunk_embeddings = model(path=model_path).encode(
        [
            f"{QUERY_PREFIX}{chunk}"
            for chunk in chunk_for_model(text=query, prefix=QUERY_PREFIX)
        ],
        batch_size=_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    pooled = chunk_embeddings.mean(axis=0)
    return pooled / np.linalg.norm(pooled)


def score_chunks(
    *, index: DenseIndex, query: str, model_path: Path | None = None
) -> np.ndarray:
    """
    :param index: A dense index built by `build_dense_index`.
    :param query: The query text - already citation-masked, see `lib.query`.
    :param model_path: The fine-tuned model the index was built with; the pretrained
        model if omitted.
    :returns: The cosine similarity of the query to every chunk, in index row order.
    """
    return index.chunk_embeddings @ _embed_query(query=query, model_path=model_path)


def retrieve(
    *, index: DenseIndex, query: str, k: int, model_path: Path | None = None
) -> list[str]:
    """
    Retrieve the top-k ruling decision_ids for `query` by cosine similarity, scoring
    each ruling by its best-matching chunk.

    :param index: A dense index built by `build_dense_index`.
    :param query: The query text - already citation-masked, see `lib.query`.
    :param k: How many results to return.
    :param model_path: The fine-tuned model the index was built with; the pretrained
        model if omitted.
    :returns: Up to `k` decision_ids, best first.
    """
    chunk_scores = score_chunks(index=index, query=query, model_path=model_path)
    # A ruling scores as its best-matching chunk. `chunk_starts` marks where each
    # ruling's run of chunk scores begins, and `reduceat` takes the max within each
    # run, giving one score per ruling, aligned with `decision_ids`.
    ruling_scores = np.maximum.reduceat(chunk_scores, index.chunk_starts)
    top_k_idx = np.argsort(-ruling_scores)[:k]
    return [index.decision_ids[i] for i in top_k_idx]
