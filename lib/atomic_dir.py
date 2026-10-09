"""Atomically materialize a cache directory."""

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_dir_write(*, final_dir: Path) -> Iterator[Path]:
    """
    Build a directory under a sibling `.tmp` path and rename it into place only
    once the caller's block exits without raising - so a process killed
    partway through never leaves a directory at `final_dir` that looks done
    but isn't.

    :param final_dir: The directory to create. Callers check that this
        doesn't exist yet themselves (that's the cache-hit condition).
    :yields: The temporary directory to populate - write into this path, not
        `final_dir`. It does not exist yet when yielded; create it yourself
        (`mkdir`, or a library call that creates it, e.g. `index.save(...)`).
    """
    tmp_dir = final_dir.with_name(f"{final_dir.name}.tmp")
    if tmp_dir.is_dir():
        shutil.rmtree(tmp_dir)
    yield tmp_dir
    tmp_dir.rename(final_dir)
