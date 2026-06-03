"""Tests for FeatureFlags."""

from __future__ import annotations

import os

import pytest

from darwin.scale.feature_flags import FeatureFlags


@pytest.fixture
def clean_env() -> None:
    keys = [
        "DARWIN_MESH_BACKEND",
        "DARWIN_RETRIEVAL_BACKEND",
        "DARWIN_RUST_KERNEL",
        "DARWIN_MULTIPROCESS",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_defaults_are_pure_python(clean_env) -> None:
    flags = FeatureFlags.read_env()
    assert flags.mesh_backend == "python"
    assert flags.retrieval_backend == "python"
    assert flags.rust_kernel is False
    assert flags.multiprocess is False


def test_torch_mesh_backend(clean_env) -> None:
    os.environ["DARWIN_MESH_BACKEND"] = "torch"
    flags = FeatureFlags.read_env()
    assert flags.mesh_backend == "torch"


def test_faiss_retrieval(clean_env) -> None:
    os.environ["DARWIN_RETRIEVAL_BACKEND"] = "faiss"
    flags = FeatureFlags.read_env()
    assert flags.retrieval_backend == "faiss"


def test_bool_env_accepts_1_yes_true(clean_env) -> None:
    for value in ("1", "true", "yes", "on", "True", "YES"):
        os.environ["DARWIN_RUST_KERNEL"] = value
        assert FeatureFlags.read_env().rust_kernel is True
    os.environ["DARWIN_RUST_KERNEL"] = "no"
    assert FeatureFlags.read_env().rust_kernel is False


def test_record_serializes(clean_env) -> None:
    flags = FeatureFlags.read_env()
    record = flags.to_record()
    assert record["mesh_backend"] == "python"
    assert "rust_kernel" in record
