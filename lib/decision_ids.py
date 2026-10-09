"""
Read/write the `decision_id` list that accompanies a cached index or
embedding directory - the row order `lib.lexical.bm25` and
`lib.dense.retrieval` align their index/embeddings against.
"""

import json
from pathlib import Path

_FILENAME = "decision_ids.json"


def read_decision_ids(*, directory: Path) -> list[str]:
    """
    :param directory: The cache directory containing `decision_ids.json`.
    :returns: The decision_ids, in on-disk order.
    """
    return json.loads((directory / _FILENAME).read_text(encoding="utf-8"))


def write_decision_ids(*, directory: Path, decision_ids: list[str]) -> None:
    """
    :param directory: The cache directory to write `decision_ids.json` into.
    :param decision_ids: The decision_ids to persist, in the order callers
        should read them back in.
    """
    (directory / _FILENAME).write_text(json.dumps(decision_ids), encoding="utf-8")
