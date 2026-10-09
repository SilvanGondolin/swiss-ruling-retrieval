"""Loading and parsing rcds/swiss_doc2doc_ir split files."""

import ast
import json
import lzma
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from dacite import Config, from_dict
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from lib.atomic_dir import atomic_dir_write
from lib.language import Language
from lib.rulings import list_ruling_ids
from paths import DATA_DIR

# Written into a split's record directory once its cases are prepared, so that
# preparation runs exactly once per local copy. Not a .json file, so it isn't a record.
_PREPARED_MARKER = ".prepared"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class Case:
    decision_id: str
    year: int
    language: Language
    facts: str
    considerations: str
    rulings: str
    chamber: str
    region: str
    origin_court: int | None
    origin_canton: int | None
    origin_chamber: int | None
    law_area: str | None
    law_sub_area: str | None
    cited_rulings: list[str]
    laws: list[str]


@dataclass(frozen=True)
class PreparedRow:
    """
    :ivar row: The row to keep, or None if the case is dropped.
    :ivar removed_citations: How many of its cited ruling ids were not in the rulings
        index.
    """

    row: dict | None
    removed_citations: int


@dataclass(frozen=True)
class PreparationSummary:
    """
    What preparing a split changed.

    :ivar kept_cases: How many cases remain.
    :ivar dropped_cases: How many cases were removed because none of their cited
        rulings is in the rulings index.
    :ivar removed_citations: How many cited ruling ids that aren't in the rulings index
        were removed, in kept and dropped cases together.
    """

    kept_cases: int
    dropped_cases: int
    removed_citations: int


def prepare_row(*, raw: dict, ruling_ids: frozenset[str]) -> PreparedRow:
    """
    Remove the cited rulings that aren't in the rulings index from a raw case row, so
    that every remaining citation can be retrieved. A case that cited rulings but has
    none left is dropped; a case that never cited any is kept unchanged.

    :param raw: One row as parsed by json.loads, straight from the split file.
    :param ruling_ids: The decision_ids of the rulings index.
    :returns: The row to keep (`raw` itself if nothing changed) and how many cited ids
        were removed.
    """
    cited = parse_id_list(raw=raw["cited_rulings"])
    resolvable = [ruling_id for ruling_id in cited if ruling_id in ruling_ids]
    removed = len(cited) - len(resolvable)
    if removed == 0:
        return PreparedRow(row=raw, removed_citations=0)
    if not resolvable:
        return PreparedRow(row=None, removed_citations=removed)
    # same stringified-list format as the source data, which `parse_id_list` reads back
    return PreparedRow(
        row={**raw, "cited_rulings": str(resolvable)}, removed_citations=removed
    )


def prepare_existing_records(
    *, records_dir: Path, ruling_ids: frozenset[str]
) -> PreparationSummary:
    """
    Apply `prepare_row` in place to a split's per-record files, for a local copy made
    before preparation existed: files of dropped cases are deleted, files that lost
    citations are rewritten. Safe to repeat or resume after an interruption, since
    preparing a prepared row changes nothing.

    :param records_dir: A split's per-record directory.
    :param ruling_ids: The decision_ids of the rulings index.
    :returns: What was changed.
    """
    kept = dropped = removed = 0
    for path in tqdm(
        sorted(records_dir.glob("*.json")), desc=f"preparing {records_dir.name}"
    ):
        raw = json.loads(path.read_text(encoding="utf-8"))
        prepared = prepare_row(raw=raw, ruling_ids=ruling_ids)
        removed += prepared.removed_citations
        if prepared.row is None:
            path.unlink()
            dropped += 1
            continue
        kept += 1
        if prepared.row is not raw:
            path.write_text(
                json.dumps(prepared.row, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    (records_dir / _PREPARED_MARKER).write_text("", encoding="utf-8")
    return PreparationSummary(
        kept_cases=kept, dropped_cases=dropped, removed_citations=removed
    )


def _print_summary(*, split: Split, summary: PreparationSummary) -> None:
    """
    :param split: The prepared split.
    :param summary: What preparing it changed.
    """
    print(
        f"prepared {split.value}: kept {summary.kept_cases} cases, dropped "
        f"{summary.dropped_cases} without a cited ruling in the rulings index, removed "
        f"{summary.removed_citations} cited ids that are not in it"
    )


def _ensure_xz_downloaded(*, split: Split) -> Path:
    """
    Download a split's .jsonl.xz from the HF hub if it's not already present locally.

    :param split: Which split to ensure is downloaded.
    :returns: Local path to the (now-guaranteed-present) .jsonl.xz file.
    """
    xz_path = DATA_DIR / "data" / f"{split}.jsonl.xz"
    if not xz_path.is_file():
        hf_hub_download(
            repo_id="rcds/swiss_doc2doc_ir",
            filename=f"data/{split}.jsonl.xz",
            repo_type="dataset",
            local_dir=DATA_DIR,
        )
    return xz_path


def ensure_data_present(*, split: Split) -> Path:
    """
    Download a split if necessary, prepare it, and write each of its JSON objects to
    its own <decision_id>.json file under data/data/<split>/. The downloaded .jsonl.xz
    is deleted once that's done - the per-record files become the canonical local copy
    from that point on.

    Preparing (see `prepare_row`) removes the cited rulings that aren't in the rulings
    index and the cases left without any, so every case's citations can be retrieved.
    A local copy made before preparation existed is prepared in place the first time
    it is loaded (its removed records can be re-downloaded by deleting the directory).
    Idempotent: returns immediately if the directory exists and is prepared. Writes
    into a temp directory and renames it into place at the end, so a process killed
    partway through doesn't leave a directory that looks done but isn't - the .xz is
    only deleted after that rename succeeds.

    :param split: Which split to materialize.
    :returns: The per-record directory path.
    """
    records_dir = DATA_DIR / "data" / split
    if records_dir.is_dir():
        if not (records_dir / _PREPARED_MARKER).is_file():
            summary = prepare_existing_records(
                records_dir=records_dir, ruling_ids=frozenset(list_ruling_ids())
            )
            _print_summary(split=split, summary=summary)
        return records_dir

    xz_path = _ensure_xz_downloaded(split=split)
    ruling_ids = frozenset(list_ruling_ids())

    kept = dropped = removed = 0
    with atomic_dir_write(final_dir=records_dir) as tmp_dir:
        tmp_dir.mkdir(parents=True)
        with lzma.open(xz_path, "rt", encoding="utf-8") as f:
            for line in f:
                prepared = prepare_row(raw=json.loads(line), ruling_ids=ruling_ids)
                removed += prepared.removed_citations
                if prepared.row is None:
                    dropped += 1
                    continue
                kept += 1
                out_path = tmp_dir / f"{prepared.row['decision_id']}.json"
                out_path.write_text(
                    json.dumps(prepared.row, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        (tmp_dir / _PREPARED_MARKER).write_text("", encoding="utf-8")

    xz_path.unlink()
    _print_summary(
        split=split,
        summary=PreparationSummary(
            kept_cases=kept, dropped_cases=dropped, removed_citations=removed
        ),
    )
    return records_dir


def parse_id_list(*, raw: str) -> list[str]:
    """Fields like `laws` and `cited_rulings` are stored as the string
    repr of a Python list, not JSON - e.g. "['a', 'b']" - so json.loads
    fails on them and they need ast.literal_eval instead.

    :param raw: The field's raw string value, e.g. "['a', 'b']" or "".
    :returns: The parsed list of ids, or an empty list for an empty string.
    """
    return ast.literal_eval(raw) if raw else []


def normalize_row(*, raw: dict) -> dict:
    """
    Fix up a raw JSON row before it reaches dacite: parse the stringified
    `laws`/`cited_rulings` lists, and turn the literal string "nan" that `law_sub_area`
    sometimes holds (a pandas-export artifact) into a real null.

    :param raw: One row as parsed by json.loads, straight from the split file.
    :returns: A copy of `raw` with `laws`/`cited_rulings` parsed to lists and
        `law_sub_area` normalized.
    """
    row = dict(raw)
    row["laws"] = parse_id_list(raw=row["laws"])
    row["cited_rulings"] = parse_id_list(raw=row["cited_rulings"])
    if row["law_sub_area"] == "nan":
        row["law_sub_area"] = None
    return row


def load_case(*, raw: dict) -> Case:
    """
    Parse one raw JSON row into a `Case`.

    Casts `year` (a JSON float like 2017.0) to int, and `language` (a plain string) to
    `Language`; every other type mismatch is left to raise, since e.g. `origin_court`
    being non-null and non-int would mean our assumptions about the schema don't hold
    on this split and that's worth knowing loudly.

    :param raw: One row as parsed by json.loads, straight from the split file.
    :returns: The row as a `Case`.
    """
    return from_dict(Case, normalize_row(raw=raw), config=Config(cast=[int, Language]))


def iter_cases(*, split: Split) -> list[Case]:
    """
    Ensure a split's per-record json files exist (downloading and expanding them
    first if needed - see `ensure_data_present`), then load all of them as `Case`s,
    sorted by decision_id for a deterministic order (a directory listing's order isn't
    otherwise guaranteed). The whole split is held in memory.

    :param split: Which split to load.
    :returns: Every record as a `Case`, sorted by decision_id.
    """
    records_dir = ensure_data_present(split=split)
    return [
        load_case(raw=json.loads(path.read_text(encoding="utf-8")))
        for path in tqdm(
            sorted(records_dir.glob("*.json")), desc=f"loading {split.value} cases"
        )
    ]
