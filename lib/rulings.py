"""Downloading and expanding the rcds/swiss_leading_decisions ruling corpus."""

import json
from dataclasses import dataclass
from pathlib import Path

from dacite import Config, from_dict
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from lib.atomic_dir import atomic_dir_write
from lib.language import Language
from paths import DATA_DIR

_JSONL_PATH = DATA_DIR / "data" / "rulings.jsonl"
_RECORDS_DIR = DATA_DIR / "data" / "rulings"


@dataclass(frozen=True)
class Ruling:
    decision_id: str
    language_id: int
    chamber_id: int
    file_id: int
    date: int | None
    topic: str
    year: int
    language: Language
    facts: str
    facts_num_tokens_bert: int
    facts_num_tokens_spacy: int
    considerations: str
    considerations_num_tokens_bert: int
    considerations_num_tokens_spacy: int
    rulings: str
    rulings_num_tokens_bert: int
    rulings_num_tokens_spacy: int
    chamber: str
    court: str
    canton: str
    region: str
    file_name: str
    html_url: str
    pdf_url: str
    file_number: str


def _ensure_rulings_downloaded() -> Path:
    """
    Download rcds/swiss_leading_decisions' rulings.jsonl from the HF hub if it's not
    already present locally.

    :returns: Local path to the (now-guaranteed-present) rulings.jsonl file.
    """
    if not _JSONL_PATH.is_file():
        hf_hub_download(
            repo_id="rcds/swiss_leading_decisions",
            filename="data/rulings.jsonl",
            repo_type="dataset",
            local_dir=DATA_DIR,
        )
    return _JSONL_PATH


def ensure_rulings_data_present() -> Path:
    """
    Download the ruling corpus if necessary, write each of its JSON objects to its.

    own <decision_id>.json file under data/data/rulings/, and delete the downloaded
    rulings.jsonl once that's done - the per-record files become the canonical
    local copy from that point on. Idempotent: returns immediately if the directory
    already exists. Writes into a temp directory and renames it into place at the
    end, so a process killed partway through doesn't leave a directory that looks
    done but isn't - the source file is only deleted after that rename succeeds.

    :returns: The per-record directory path.
    """
    if _RECORDS_DIR.is_dir():
        return _RECORDS_DIR

    jsonl_path = _ensure_rulings_downloaded()

    with atomic_dir_write(final_dir=_RECORDS_DIR) as tmp_dir:
        tmp_dir.mkdir(parents=True)
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                out_path = tmp_dir / f"{row['decision_id']}.json"
                out_path.write_text(
                    json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
                )

    jsonl_path.unlink()
    return _RECORDS_DIR


def list_ruling_ids() -> list[str]:
    """
    :returns: Every ruling's decision_id, sorted - read from the per-record file
        names, without parsing the files.
    """
    return sorted(p.stem for p in ensure_rulings_data_present().glob("*.json"))


def iter_rulings(*, ruling_ids: frozenset[str] | None = None) -> list[Ruling]:
    """
    Ensure the ruling corpus's per-record json files exist (downloading and
    expanding them first if needed - see `ensure_rulings_data_present`), then load
    all of them as `Ruling`s, sorted by decision_id for a deterministic order (a
    directory listing's order isn't otherwise guaranteed). The whole corpus is held
    in memory.

    :param ruling_ids: Only load these rulings (a reduced corpus, for fast
        iteration); all of them if omitted.
    :returns: Every requested ruling, sorted by decision_id.
    """
    records_dir = ensure_rulings_data_present()
    return [
        from_dict(
            Ruling,
            json.loads(path.read_text(encoding="utf-8")),
            config=Config(cast=[int, Language]),
        )
        for path in tqdm(sorted(records_dir.glob("*.json")), desc="loading rulings")
        if ruling_ids is None or path.stem in ruling_ids
    ]
