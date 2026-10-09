from lib.decision_ids import read_decision_ids, write_decision_ids


def test_read_decision_ids_returns_what_write_decision_ids_wrote(tmp_path):
    write_decision_ids(directory=tmp_path, decision_ids=["a", "b", "c"])
    assert read_decision_ids(directory=tmp_path) == ["a", "b", "c"]
