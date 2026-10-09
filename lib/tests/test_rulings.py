import json
from pathlib import Path
from unittest.mock import patch

from lib.rulings import Ruling, iter_rulings, list_ruling_ids

_BASE_ROW = {
    "decision_id": "placeholder",
    "language_id": 1,
    "chamber_id": 1,
    "file_id": 1,
    "date": 788918400000,
    "topic": "",
    "year": 2020.0,
    "language": "de",
    "facts": "Sachverhalt",
    "facts_num_tokens_bert": 10,
    "facts_num_tokens_spacy": 8,
    "considerations": "Erwägungen",
    "considerations_num_tokens_bert": 20,
    "considerations_num_tokens_spacy": 15,
    "rulings": "Urteil",
    "rulings_num_tokens_bert": 5,
    "rulings_num_tokens_spacy": 4,
    "chamber": "CH_BGE_001",
    "court": "CH_BGE",
    "canton": "CH",
    "region": "Federation",
    "file_name": "file",
    "html_url": "https://example.com",
    "pdf_url": "",
    "file_number": "BGE_1_I_1",
}


def _write_ruling_json(*, path: Path, decision_id: str, **overrides) -> None:
    row = {**_BASE_ROW, "decision_id": decision_id, **overrides}
    path.write_text(json.dumps(row), encoding="utf-8")


def test_iter_rulings_yields_ruling_objects_sorted_by_decision_id(tmp_path):
    _write_ruling_json(path=tmp_path / "b.json", decision_id="b-id")
    _write_ruling_json(path=tmp_path / "a.json", decision_id="a-id")

    with patch("lib.rulings.ensure_rulings_data_present", return_value=tmp_path):
        rulings = list(iter_rulings())

    assert [r.decision_id for r in rulings] == ["a-id", "b-id"]
    assert all(isinstance(r, Ruling) for r in rulings)


def test_iter_rulings_casts_year_to_int(tmp_path):
    _write_ruling_json(path=tmp_path / "a.json", decision_id="a-id", year=1995.0)

    with patch("lib.rulings.ensure_rulings_data_present", return_value=tmp_path):
        [ruling] = list(iter_rulings())

    assert ruling.year == 1995
    assert isinstance(ruling.year, int)


def test_iter_rulings_handles_null_date(tmp_path):
    _write_ruling_json(path=tmp_path / "a.json", decision_id="a-id", date=None)

    with patch("lib.rulings.ensure_rulings_data_present", return_value=tmp_path):
        [ruling] = list(iter_rulings())

    assert ruling.date is None


def test_iter_rulings_only_loads_the_requested_ruling_ids(tmp_path):
    for decision_id in ("a-id", "b-id", "c-id"):
        _write_ruling_json(
            path=tmp_path / f"{decision_id}.json", decision_id=decision_id
        )

    with patch("lib.rulings.ensure_rulings_data_present", return_value=tmp_path):
        rulings = iter_rulings(ruling_ids=frozenset({"c-id", "a-id", "missing-id"}))

    assert [r.decision_id for r in rulings] == ["a-id", "c-id"]


def test_list_ruling_ids_returns_sorted_ids_without_parsing_the_files(tmp_path):
    (tmp_path / "b-id.json").write_text("not valid json", encoding="utf-8")
    (tmp_path / "a-id.json").write_text("not valid json", encoding="utf-8")

    with patch("lib.rulings.ensure_rulings_data_present", return_value=tmp_path):
        assert list_ruling_ids() == ["a-id", "b-id"]
