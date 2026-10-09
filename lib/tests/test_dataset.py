import json
import lzma
from pathlib import Path

import lib.dataset as dataset
from lib.dataset import (
    Split,
    ensure_data_present,
    parse_id_list,
    prepare_existing_records,
    prepare_row,
)

_IDS = frozenset({"a", "b"})


def _row(*, decision_id: str, cited: list[str] | None) -> dict:
    # `cited_rulings` is a stringified list in the source data, "" when empty
    return {
        "decision_id": decision_id,
        "cited_rulings": str(cited) if cited else "",
    }


def test_prepare_row_keeps_a_row_whose_citations_all_resolve_unchanged():
    raw = _row(decision_id="c", cited=["a", "b"])
    prepared = prepare_row(raw=raw, ruling_ids=_IDS)
    assert prepared.row is raw
    assert prepared.removed_citations == 0


def test_prepare_row_removes_only_the_citations_missing_from_the_index():
    prepared = prepare_row(
        raw=_row(decision_id="c", cited=["a", "x", "b", "y"]), ruling_ids=_IDS
    )
    assert prepared.removed_citations == 2
    assert prepared.row is not None
    assert parse_id_list(raw=prepared.row["cited_rulings"]) == ["a", "b"]


def test_prepare_row_does_not_modify_the_row_it_is_given():
    raw = _row(decision_id="c", cited=["a", "x"])
    prepare_row(raw=raw, ruling_ids=_IDS)
    assert parse_id_list(raw=raw["cited_rulings"]) == ["a", "x"]


def test_prepare_row_drops_a_case_with_no_citation_left():
    prepared = prepare_row(raw=_row(decision_id="c", cited=["x", "y"]), ruling_ids=_IDS)
    assert prepared.row is None
    assert prepared.removed_citations == 2


def test_prepare_row_keeps_a_case_that_never_cited_anything():
    raw = _row(decision_id="c", cited=None)
    prepared = prepare_row(raw=raw, ruling_ids=_IDS)
    assert prepared.row is raw
    assert prepared.removed_citations == 0


def test_prepare_row_keeps_the_order_of_the_remaining_citations():
    prepared = prepare_row(
        raw=_row(decision_id="c", cited=["b", "x", "a"]), ruling_ids=_IDS
    )
    assert prepared.row is not None
    assert parse_id_list(raw=prepared.row["cited_rulings"]) == ["b", "a"]


def _write_records(*, records_dir: Path, rows: list[dict]) -> None:
    records_dir.mkdir(parents=True)
    for row in rows:
        (records_dir / f"{row['decision_id']}.json").write_text(json.dumps(row))


_MIXED_ROWS = [
    _row(decision_id="keep", cited=["a"]),
    _row(decision_id="trim", cited=["a", "x"]),
    _row(decision_id="drop", cited=["x"]),
    _row(decision_id="none", cited=None),
]


def test_prepare_existing_records_deletes_and_rewrites_files_in_place(tmp_path):
    records_dir = tmp_path / "train"
    _write_records(records_dir=records_dir, rows=_MIXED_ROWS)

    summary = prepare_existing_records(records_dir=records_dir, ruling_ids=_IDS)

    assert sorted(p.stem for p in records_dir.glob("*.json")) == [
        "keep",
        "none",
        "trim",
    ]
    trimmed = json.loads((records_dir / "trim.json").read_text())
    assert parse_id_list(raw=trimmed["cited_rulings"]) == ["a"]
    assert (summary.kept_cases, summary.dropped_cases, summary.removed_citations) == (
        3,
        1,
        2,
    )


def test_prepare_existing_records_marks_the_directory_prepared_and_is_repeatable(
    tmp_path,
):
    records_dir = tmp_path / "train"
    _write_records(records_dir=records_dir, rows=_MIXED_ROWS)
    prepare_existing_records(records_dir=records_dir, ruling_ids=_IDS)
    assert (records_dir / ".prepared").is_file()

    again = prepare_existing_records(records_dir=records_dir, ruling_ids=_IDS)

    assert (again.kept_cases, again.dropped_cases, again.removed_citations) == (3, 0, 0)


def _patch_data(monkeypatch, *, data_dir: Path) -> None:
    monkeypatch.setattr(dataset, "DATA_DIR", data_dir)
    monkeypatch.setattr(dataset, "list_ruling_ids", lambda: sorted(_IDS))


def test_ensure_data_present_prepares_a_fresh_download(tmp_path, monkeypatch):
    _patch_data(monkeypatch, data_dir=tmp_path)
    (tmp_path / "data").mkdir()
    xz_path = tmp_path / "data" / "train.jsonl.xz"
    with lzma.open(xz_path, "wt", encoding="utf-8") as f:
        for row in _MIXED_ROWS:
            f.write(json.dumps(row) + "\n")

    records_dir = ensure_data_present(split=Split.TRAIN)

    assert sorted(p.stem for p in records_dir.glob("*.json")) == [
        "keep",
        "none",
        "trim",
    ]
    assert (records_dir / ".prepared").is_file()
    assert not xz_path.exists()


def test_ensure_data_present_prepares_an_existing_unprepared_copy_once(
    tmp_path, monkeypatch
):
    _patch_data(monkeypatch, data_dir=tmp_path)
    records_dir = tmp_path / "data" / "train"
    _write_records(records_dir=records_dir, rows=_MIXED_ROWS)

    ensure_data_present(split=Split.TRAIN)
    assert sorted(p.stem for p in records_dir.glob("*.json")) == [
        "keep",
        "none",
        "trim",
    ]

    # a prepared directory is returned as is, without looking at the rulings index again
    monkeypatch.setattr(
        dataset, "list_ruling_ids", lambda: (_ for _ in ()).throw(AssertionError)
    )
    assert ensure_data_present(split=Split.TRAIN) == records_dir
