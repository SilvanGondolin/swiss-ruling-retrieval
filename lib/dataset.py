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
from paths import DATA_DIR


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
    Download a split if necessary, write each of its JSON objects to its own.

    <decision_id>.json file under data/data/<split>/, and delete the downloaded
    .jsonl.xz once that's done - the per-record files become the canonical local
    copy from that point on. Idempotent: returns immediately if the directory
    already exists. Writes into a temp directory and renames it into place at the
    end, so a process killed partway through doesn't leave a directory that looks
    done but isn't - the .xz is only deleted after that rename succeeds.

    :param split: Which split to materialize.
    :returns: The per-record directory path.
    """
    records_dir = DATA_DIR / "data" / split
    if records_dir.is_dir():
        return records_dir

    xz_path = _ensure_xz_downloaded(split=split)

    with atomic_dir_write(final_dir=records_dir) as tmp_dir:
        tmp_dir.mkdir(parents=True)
        with lzma.open(xz_path, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                out_path = tmp_dir / f"{row['decision_id']}.json"
                out_path.write_text(
                    json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
                )

    xz_path.unlink()
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
    being non-null and non-int would mean our assumptions about the schema don't hold on
    this split and that's worth knowing loudly.

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
