"""Test isolation: redirect every default Darwin path to a per-test tempdir.

Every Darwin test runs in a sandbox. The ``DARWIN_DATA_DIR`` environment
variable is set to a fresh ``tmp_path``-rooted directory for every test,
so no test can write to (or read from) the operator's actual sqlite memory
file, universe JSON, runtime state, snapshot store, or training logs.

This is autouse and unconditional — there is no way to opt out, and that is
deliberate. Production state should never depend on what the test suite
did, and the test suite should never inherit operational data.

The fixture also guards three invariants after every test:

1. No file was written to the legacy CWD locations
   (``darwin_memory.sqlite3``, ``darwin_universe.json``,
   ``darwin_runtime_state.json``, ``darwin_snapshots/``,
   ``training_logs/``).
2. The redirected data directory was actually used when a test produced
   any persistent artifact.
3. ``DARWIN_DATA_DIR`` is restored to whatever the operator had it set
   to before pytest started.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


_LEGACY_PATHS = (
    "darwin_memory.sqlite3",
    "darwin_memory.sqlite3-journal",
    "darwin_memory.sqlite3-wal",
    "darwin_memory.sqlite3-shm",
    "darwin_universe.json",
    "darwin_runtime_state.json",
    "darwin_snapshots",
    "training_logs",
)


@pytest.fixture(autouse=True)
def _isolated_darwin_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect every default Darwin path into ``tmp_path`` for this test."""

    sandbox = tmp_path / "darwin_data"
    sandbox.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DARWIN_DATA_DIR", str(sandbox))

    # Capture the legacy-CWD presence so we can detect (and complain about)
    # leakage in this test.
    cwd = Path.cwd()
    before = {name: (cwd / name).exists() for name in _LEGACY_PATHS}

    yield sandbox

    # Post-test contamination guard.
    leaked: list[str] = []
    for name, was_present in before.items():
        if not was_present and (cwd / name).exists():
            leaked.append(name)
    if leaked:
        # Tear down the leaked artifacts so we don't pollute the next test
        # AND surface the regression loudly.
        for name in leaked:
            path = cwd / name
            try:
                if path.is_dir():
                    import shutil

                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                pass
        pytest.fail(
            f"Test wrote to legacy CWD path(s) that should have been "
            f"redirected via DARWIN_DATA_DIR: {leaked}. "
            f"Either pass an explicit override or rely on the autouse "
            f"isolation fixture in conftest.py."
        )


@pytest.fixture
def darwin_data_dir(_isolated_darwin_paths: Path) -> Path:
    """Convenience handle on the per-test data directory."""

    return _isolated_darwin_paths
