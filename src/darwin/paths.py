"""Centralized resolution for Darwin's persistent-state paths.

A single environment variable — ``DARWIN_DATA_DIR`` — controls where every
persistent artifact lands: the sqlite memory file, the concept universe
JSON, the runtime-state snapshot, the snapshot store directory, the
training-logs directory. Production code defaults to the current working
directory (the legacy v5 behavior). Tests set ``DARWIN_DATA_DIR`` to a
per-test temp directory so they cannot contaminate operational state.

The paths are resolved *lazily at call time*. Every consumer either takes
an explicit override (preferred for libraries) or calls one of the helper
functions below with no arguments. The helper reads ``DARWIN_DATA_DIR``
at the moment it is called, so tests that change the env var between
runs see the new value.
"""

from __future__ import annotations

import os
from pathlib import Path


DATA_DIR_ENV = "DARWIN_DATA_DIR"


def data_dir() -> Path:
    """The root directory for every persistent artifact."""

    raw = os.environ.get(DATA_DIR_ENV)
    if raw:
        return Path(raw).expanduser()
    return Path.cwd()


def memory_path() -> Path:
    return data_dir() / "darwin_memory.sqlite3"


def universe_path() -> Path:
    return data_dir() / "darwin_universe.json"


def runtime_state_path() -> Path:
    return data_dir() / "darwin_runtime_state.json"


def snapshots_dir() -> Path:
    return data_dir() / "darwin_snapshots"


def training_logs_dir() -> Path:
    return data_dir() / "training_logs"


def plan_log_path() -> Path:
    return training_logs_dir() / "plans.jsonl"


def background_log_path() -> Path:
    return training_logs_dir() / "background.jsonl"


def metrics_log_path() -> Path:
    return training_logs_dir() / "metrics.jsonl"


def dlm_training_pairs_path() -> Path:
    return training_logs_dir() / "dlm_training_pairs.jsonl"


def generated_modules_dir() -> Path:
    """Where Darwin writes its self-generated modules."""

    return Path(__file__).resolve().parent / "generated"


__all__ = [
    "DATA_DIR_ENV",
    "background_log_path",
    "data_dir",
    "dlm_training_pairs_path",
    "generated_modules_dir",
    "memory_path",
    "metrics_log_path",
    "plan_log_path",
    "runtime_state_path",
    "snapshots_dir",
    "training_logs_dir",
    "universe_path",
]
