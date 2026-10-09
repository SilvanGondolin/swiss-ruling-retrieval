import statistics
from collections import Counter

import cyclopts
from tqdm import tqdm

from lib.dataset import Split, iter_cases


def inspect(*, split: Split = Split.VALIDATION) -> None:
    """
    Print schema and summary statistics for a rcds/swiss_doc2doc_ir split, downloading
    it first if it's not already present locally.

    :param split: Which split to inspect.
    """
    languages: Counter[str] = Counter()
    law_areas: Counter[str | None] = Counter()
    n_laws: list[int] = []
    n_cited_rulings: list[int] = []
    considerations_len: list[int] = []
    facts_len: list[int] = []
    empty_facts = 0
    null_origin_court = 0
    total = 0

    for case in tqdm(iter_cases(split=split)):
        total += 1
        languages[case.language] += 1
        law_areas[case.law_area] += 1
        n_laws.append(len(case.laws))
        n_cited_rulings.append(len(case.cited_rulings))
        considerations_len.append(len(case.considerations))
        facts_len.append(len(case.facts))
        if not case.facts:
            empty_facts += 1
        if case.origin_court is None:
            null_origin_court += 1

    def stats(xs: list[int]) -> str:
        return (
            f"mean={statistics.mean(xs):.1f} median={statistics.median(xs)} "
            f"min={min(xs)} max={max(xs)}"
        )

    print(f"total records: {total}")
    print(f"languages: {dict(languages)}")
    print(f"law_area (top 10): {law_areas.most_common(10)}")
    print(f"laws cited per case: {stats(n_laws)}")
    print(f"cited_rulings per case: {stats(n_cited_rulings)}")
    print(f"considerations length (chars): {stats(considerations_len)}")
    print(f"facts length (chars): {stats(facts_len)}")
    print(f"empty facts: {empty_facts}/{total} ({100 * empty_facts / total:.1f}%)")
    print(
        f"null origin_court: {null_origin_court}/{total} "
        f"({100 * null_origin_court / total:.1f}%)"
    )


if __name__ == "__main__":
    cyclopts.run(inspect)
