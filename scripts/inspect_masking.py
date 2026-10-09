"""Show which text spans mask_citations() actually replaces, on real data."""

import csv
import random

import cyclopts
from tqdm import tqdm

from lib.dataset import Split, iter_cases
from lib.masking import find_citations
from paths import DATA_DIR


def inspect_masking(*, split: Split = Split.VALIDATION, preview: int = 20) -> None:
    """
    Scan a split, write every citation span mask_citations() would replace to a CSV,
    then interactively show random batches of them - for spot-checking false
    positives/negatives without reading the whole (often huge) CSV.

    :param split: Which split to scan.
    :param preview: How many rows to show per batch, and how many more to offer
        showing each time.
    """
    out_path = DATA_DIR / f"masking_inspection_{split}.csv"

    rows: list[dict[str, str]] = []
    for case in tqdm(iter_cases(split=split), desc=f"scanning {split}"):
        for field_name, text in {
            "facts": case.facts,
            "considerations": case.considerations,
        }.items():
            for citation in find_citations(text=text):
                rows.append(
                    {
                        "decision_id": case.decision_id,
                        "language": case.language,
                        "field": field_name,
                        "label": citation.label,
                        "snippet": citation.text,
                    }
                )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["decision_id", "language", "field", "label", "snippet"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} citation spans found, written to {out_path}")
    print()

    shuffled = rows.copy()
    random.shuffle(shuffled)
    _show_random_batches(shuffled_rows=shuffled, batch_size=preview)


def _show_random_batches(*, shuffled_rows: list[dict], batch_size: int) -> None:
    """
    Print `shuffled_rows` in batches of `batch_size`, asking after each one whether
    to show another - stops once the user declines or the rows run out.

    :param shuffled_rows: Rows already in random order, so consecutive batches are
        distinct random samples rather than repeats.
    :param batch_size: How many rows to print per batch, and to ask about each time.
    """
    offset = 0
    while offset < len(shuffled_rows):
        batch = shuffled_rows[offset : offset + batch_size]
        for row in batch:
            print(
                f"{row['decision_id']}  {row['language']}  {row['field']:14}  "
                f"{row['label']:7}  {row['snippet']!r}"
            )
        offset += len(batch)

        if offset >= len(shuffled_rows):
            print(f"\n(no more rows - {offset} shown)")
            break

        answer = input(f"\nShow {batch_size} more? (y/n) ").strip().lower()
        if not answer.startswith("y"):
            break


if __name__ == "__main__":
    cyclopts.run(inspect_masking)
