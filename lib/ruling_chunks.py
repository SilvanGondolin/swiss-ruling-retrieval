"""The ruling corpus split into embedding-model-sized chunks."""

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from lib.chunking import chunk_starts
from lib.dense.embedding_model import PASSAGE_PREFIX, chunk_for_model
from lib.language import Language
from lib.rulings import iter_rulings


@dataclass(frozen=True)
class RulingChunks:
    """
    Every chunk of every ruling's `considerations`, stacked in ruling order - the
    row order of both the dense and the chunked BM25 index.

    :ivar decision_ids: The ruling for each group of chunks.
    :ivar languages: Each ruling's language, aligned with `decision_ids`.
    :ivar chunk_starts: The row in `texts` where each ruling's chunks begin, aligned
        with `decision_ids`; a ruling's chunks run until the next ruling's start.
    :ivar texts: The chunk texts, without any model prefix.
    """

    decision_ids: list[str]
    languages: list[Language]
    chunk_starts: np.ndarray
    texts: list[str]


def load_ruling_chunks(*, ruling_ids: frozenset[str] | None = None) -> RulingChunks:
    """
    Chunk the whole ruling corpus's `considerations` text, downloading/expanding the
    corpus first if necessary (see `lib.rulings`). Deterministic, so every index built
    from it lines up row for row.

    :param ruling_ids: Only chunk these rulings (a reduced corpus, for fast iteration);
        all of them if omitted.
    :returns: The chunks of every ruling.
    """
    decision_ids: list[str] = []
    languages: list[Language] = []
    chunk_counts: list[int] = []
    texts: list[str] = []
    for ruling in tqdm(iter_rulings(ruling_ids=ruling_ids), desc="chunking rulings"):
        chunks = chunk_for_model(text=ruling.considerations, prefix=PASSAGE_PREFIX)
        decision_ids.append(ruling.decision_id)
        languages.append(ruling.language)
        chunk_counts.append(len(chunks))
        texts.extend(chunks)
    return RulingChunks(
        decision_ids=decision_ids,
        languages=languages,
        chunk_starts=chunk_starts(chunk_counts=chunk_counts),
        texts=texts,
    )
