"""Mine (query chunk, ruling chunk) training pairs from the train split's citations,
using the chunked BM25 index's best guess at which chunks match (see
`lib.finetuning.pair_mining`)."""

import time

import cyclopts

from lib.dataset import Split, iter_cases
from lib.finetuning.pair_mining import mine_pairs, pairs_name, save_pairs
from lib.lexical.bm25_chunked import build_chunk_index
from lib.ruling_chunks import load_ruling_chunks
from lib.sampling import sample_items


def mine(
    *, train_cases: int | None = None, seed: int = 0, limit: int | None = None
) -> None:
    """
    Mine and cache training pairs, or - with `limit` - only time a sample and print a
    few of its pairs, writing nothing. Always mines against the full ruling corpus, so
    a positive can be any cited ruling.

    :param train_cases: Mine only this many sampled train cases (with citations), for
        fast iteration; all of them if omitted.
    :param seed: The sampling seed. Training (`scripts.train`) must use the same
        `train_cases` and `seed` to find these pairs.
    :param limit: Only mine the first this-many of those cases, as a dry run.
    """
    chunk_index = build_chunk_index()
    ruling_chunks = load_ruling_chunks()
    cases = sample_items(
        items=[c for c in iter_cases(split=Split.TRAIN) if c.cited_rulings],
        n=train_cases,
        seed=seed,
    )
    if limit is None:
        pairs = mine_pairs(
            cases=cases, chunk_index=chunk_index, ruling_chunks=ruling_chunks
        )
        name = pairs_name(train_cases=train_cases, seed=seed)
        print(f"wrote {save_pairs(name=name, pairs=pairs)}")
        return

    start = time.perf_counter()
    pairs_sample = list(
        mine_pairs(
            cases=cases[:limit], chunk_index=chunk_index, ruling_chunks=ruling_chunks
        )
    )
    elapsed = time.perf_counter() - start
    zero = sum(p.bm25_score == 0.0 for p in pairs_sample)
    print(
        f"{limit} cases -> {len(pairs_sample)} pairs in {elapsed:.1f}s ({zero} score 0)"
    )
    print(f"estimated {elapsed / limit:.3f}s/case")
    for pair in pairs_sample[:2]:
        print(f"\n[{pair.case_id} -> {pair.ruling_id}] score {pair.bm25_score:.1f}")
        print("QUERY :", pair.query_chunk[:300].replace("\n", " "))
        print("RULING:", pair.ruling_chunk[:300].replace("\n", " "))


if __name__ == "__main__":
    cyclopts.run(mine)
