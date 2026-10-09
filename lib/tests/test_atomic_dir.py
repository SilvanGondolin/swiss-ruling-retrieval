from lib.atomic_dir import atomic_dir_write


def test_atomic_dir_write_renames_tmp_dir_into_place_on_success(tmp_path):
    final_dir = tmp_path / "cache"
    with atomic_dir_write(final_dir=final_dir) as working_dir:
        working_dir.mkdir()
        (working_dir / "file.txt").write_text("content", encoding="utf-8")

    assert final_dir.is_dir()
    assert (final_dir / "file.txt").read_text(encoding="utf-8") == "content"
    assert not working_dir.exists()


def test_atomic_dir_write_cleans_up_a_stale_tmp_dir_from_a_previous_failed_run(
    tmp_path,
):
    final_dir = tmp_path / "cache"
    stale_tmp_dir = tmp_path / "cache.tmp"
    stale_tmp_dir.mkdir()
    (stale_tmp_dir / "leftover.txt").write_text("stale", encoding="utf-8")

    with atomic_dir_write(final_dir=final_dir) as working_dir:
        working_dir.mkdir()

    assert not (final_dir / "leftover.txt").exists()
