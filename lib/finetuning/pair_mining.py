"""
Mining (query chunk, ruling chunk) training pairs for contrastive fine-tuning.

A training example is a (case, cited ruling) citation, but both sides are many 512-token
chunks and nothing says which chunks the citation is "about". The chunked BM25 index's
best guess stands in for that: for each citation, the pair of a query chunk and a chunk
of the cited ruling with the highest BM25 score.
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lib.atomic_dir import atomic_dir_write
from lib.dataset import Case
from lib.dense.embedding_model import QUERY_PREFIX, chunk_for_model
from lib.lexical.bm25_chunked import ChunkIndex, score_chunks
from lib.query import build_query
from lib.ruling_chunks import RulingChunks
from paths import DATA_DIR

_PAIRS_FILENAME = "pairs.jsonl"


@dataclass(frozen=True)
class TrainingPair:
    """
    :ivar case_id: The citing case's decision_id.
    :ivar ruling_id: The cited ruling's decision_id.
    :ivar query_chunk: The chosen chunk of the case's query, without model prefix.
    :ivar ruling_chunk: The chosen chunk of the cited ruling, without model prefix.
    :ivar bm25_score: The pair's BM25 score; 0.0 means no lexical overlap at all, so
        the choice of chunks was arbitrary.
    """

    case_id: str
    ruling_id: str
    query_chunk: str
    ruling_chunk: str
    bm25_score: float


@dataclass(frozen=True)
class ChunkPair:
    """
    :ivar query_chunk_index: Which of the query's chunks.
    :ivar ruling_chunk_index: Which of the ruling's chunks (index within the ruling).
    :ivar score: The pair's BM25 score.
    """

    query_chunk_index: int
    ruling_chunk_index: int
    score: float


def best_chunk_pair(*, chunk_scores: Sequence[np.ndarray]) -> ChunkPair:
    """
    :param chunk_scores: One array per query chunk: that chunk's BM25 score against
        each chunk of a single ruling.
    :returns: The best-scoring query chunk and ruling chunk. Ties go to the earliest
        chunk.
    """
    best = ChunkPair(query_chunk_index=0, ruling_chunk_index=0, score=float("-inf"))
    for query_chunk_index, scores in enumerate(chunk_scores):
        ruling_chunk_index = int(np.argmax(scores))
        score = float(scores[ruling_chunk_index])
        if score > best.score:
            best = ChunkPair(
                query_chunk_index=query_chunk_index,
                ruling_chunk_index=ruling_chunk_index,
                score=score,
            )
    return best


def mine_pairs(
    *,
    cases: Iterable[Case],
    chunk_index: ChunkIndex,
    ruling_chunks: RulingChunks,
) -> Iterable[TrainingPair]:
    """
    Yield one pair per (case, cited ruling) citation whose ruling is in the corpus.

    :param cases: The citing cases.
    :param chunk_index: The chunked BM25 index over `ruling_chunks`.
    :param ruling_chunks: The chunked ruling corpus the index was built from.
    :returns: The best chunk pair of each resolvable citation.
    """
    row_of = {d: i for i, d in enumerate(ruling_chunks.decision_ids)}
    ends = np.append(ruling_chunks.chunk_starts[1:], len(ruling_chunks.texts))
    for case in cases:
        cited = [r for r in case.cited_rulings if r in row_of]
        if not cited:
            continue
        query_chunks = chunk_for_model(text=build_query(case=case), prefix=QUERY_PREFIX)
        all_scores = [
            score_chunks(chunk_index=chunk_index, query=chunk, language=case.language)
            for chunk in query_chunks
        ]
        for ruling_id in cited:
            row = row_of[ruling_id]
            start, end = ruling_chunks.chunk_starts[row], ends[row]
            best = best_chunk_pair(
                chunk_scores=[scores[start:end] for scores in all_scores]
            )
            yield TrainingPair(
                case_id=case.decision_id,
                ruling_id=ruling_id,
                query_chunk=query_chunks[best.query_chunk_index],
                ruling_chunk=ruling_chunks.texts[start + best.ruling_chunk_index],
                bm25_score=best.score,
            )


def pairs_name(*, train_cases: int | None, seed: int) -> str:
    """
    :param train_cases: How many train cases were sampled to mine pairs from; None for
        all of them.
    :param seed: The sampling seed.
    :returns: The name that identifies those mined pairs on disk.
    """
    return "train" if train_cases is None else f"train_n{train_cases}_s{seed}"


def _pairs_dir(*, name: str) -> Path:
    """
    :param name: What the pairs were mined from, e.g. a split or a dev subset tag.
    :returns: The cache directory for those mined pairs.
    """
    return DATA_DIR / "training_pairs" / name


def save_pairs(*, name: str, pairs: Iterable[TrainingPair]) -> Path:
    """
    Write `pairs` to disk as JSON lines, atomically (see `lib.atomic_dir`).

    :param name: What the pairs were mined from, e.g. a split or a dev subset tag.
    :param pairs: The pairs to persist.
    :returns: The cache directory the pairs were written to.
    """
    final_dir = _pairs_dir(name=name)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    with atomic_dir_write(final_dir=final_dir) as tmp_dir:
        tmp_dir.mkdir(parents=True)
        with (tmp_dir / _PAIRS_FILENAME).open("w", encoding="utf-8") as f:
            for pair in tqdm(pairs, desc=f"mining {name} pairs"):
                f.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")
    return final_dir


def load_pairs(*, name: str) -> list[TrainingPair]:
    """
    :param name: What the pairs were mined from, as given to `save_pairs`.
    :returns: The pairs `save_pairs` wrote under that name.
    """
    path = _pairs_dir(name=name) / _PAIRS_FILENAME
    with path.open(encoding="utf-8") as f:
        return [TrainingPair(**json.loads(line)) for line in f]
