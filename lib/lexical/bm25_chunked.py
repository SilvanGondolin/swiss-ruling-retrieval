"""
BM25 retrieval over ruling chunks, row-aligned with the dense index.

Indexes exactly the chunks `lib.ruling_chunks` produces - the same texts the dense
model embeds - so a chunk's BM25 score and its dense score refer to the same passage.
Chunks are roughly equal in length, so BM25's length normalization stops penalizing
long rulings the way it does in the whole-document index (`lib.lexical.bm25`). A ruling scores
as its best-matching chunk, like in `lib.dense.retrieval`.
"""

from dataclasses import dataclass

import bm25s
import numpy as np
from tqdm import tqdm

from lib.atomic_dir import atomic_dir_write
from lib.decision_ids import read_decision_ids, write_decision_ids
from lib.index_cache import cache_name
from lib.language import Language
from lib.lexical.bm25 import tokenize
from lib.ruling_chunks import load_ruling_chunks
from paths import INDEX_DIR

_CACHE_BASE = "bm25_chunk_index"


@dataclass(frozen=True)
class ChunkIndex:
    """
    A BM25 index with one document per ruling chunk.

    :ivar index: The BM25 index, one document per chunk.
    :ivar chunk_starts: The row (document) where each ruling's chunks begin, aligned
        with `decision_ids`; a ruling's chunks run until the next ruling's start.
    :ivar decision_ids: The ruling for each group of chunks.
    """

    index: bm25s.BM25
    chunk_starts: np.ndarray
    decision_ids: list[str]


def build_chunk_index(*, ruling_ids: frozenset[str] | None = None) -> ChunkIndex:
    """
    Build a BM25 index over every chunk of the ruling corpus's `considerations` text,
    tokenizing each chunk with its ruling's language. Cached to disk under.

    data/indexes/ (one directory per distinct set of rulings, see `lib.index_cache`)
    - idempotent, only built once.

    :param ruling_ids: Only index these rulings (a reduced corpus, for fast
        iteration); all of them if omitted.
    :returns: The index, with `decision_ids` aligned to the chunk groups.
    """
    index_dir = INDEX_DIR / cache_name(base=_CACHE_BASE, ruling_ids=ruling_ids)
    if index_dir.is_dir():
        return ChunkIndex(
            index=bm25s.BM25.load(
                str(index_dir), load_corpus=False, show_progress=False
            ),
            chunk_starts=np.load(index_dir / "chunk_starts.npy"),
            decision_ids=read_decision_ids(directory=index_dir),
        )

    chunks = load_ruling_chunks(ruling_ids=ruling_ids)
    chunk_counts = np.diff(np.append(chunks.chunk_starts, len(chunks.texts)))
    chunk_languages = np.repeat(np.array(chunks.languages), chunk_counts)
    corpus_tokens = [
        tokenize(text, Language(language))
        for text, language in tqdm(
            zip(chunks.texts, chunk_languages, strict=True),
            total=len(chunks.texts),
            desc="tokenizing chunks for BM25",
        )
    ]

    index = bm25s.BM25()
    index.index(corpus_tokens, show_progress=False)

    with atomic_dir_write(final_dir=index_dir) as tmp_dir:
        index.save(str(tmp_dir), show_progress=False)
        np.save(tmp_dir / "chunk_starts.npy", chunks.chunk_starts)
        write_decision_ids(directory=tmp_dir, decision_ids=chunks.decision_ids)

    return ChunkIndex(
        index=index,
        chunk_starts=chunks.chunk_starts,
        decision_ids=chunks.decision_ids,
    )


def score_chunks(
    *, chunk_index: ChunkIndex, query: str, language: Language
) -> np.ndarray:
    """
    :param chunk_index: An index built by `build_chunk_index`.
    :param query: The query text - already citation-masked, see `lib.query`.
    :param language: The query's language.
    :returns: The BM25 score of every chunk, in index row order.
    """
    query_tokens = tokenize(query, language)
    if not query_tokens:
        return np.zeros(int(chunk_index.index.scores["num_docs"]))
    return chunk_index.index.get_scores(query_tokens)


def retrieve(
    *, chunk_index: ChunkIndex, query: str, language: Language, k: int
) -> list[str]:
    """
    Retrieve the top-k ruling decision_ids for `query`, scoring each ruling by its best-
    matching chunk.

    :param chunk_index: An index built by `build_chunk_index`.
    :param query: The query text - already citation-masked, see `lib.query`.
    :param language: The query's language.
    :param k: How many results to return.
    :returns: Up to `k` decision_ids, best first.
    """
    chunk_scores = score_chunks(chunk_index=chunk_index, query=query, language=language)
    ruling_scores = np.maximum.reduceat(chunk_scores, chunk_index.chunk_starts)
    top_k_idx = np.argsort(-ruling_scores)[:k]
    return [chunk_index.decision_ids[i] for i in top_k_idx]
