"""Tests for V-Scale wiring into DarwinRuntime."""

from __future__ import annotations

import os

import pytest

from darwin.agent import Darwin
from darwin.embodiment import RoomSimulationAdapter
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.scale.feature_flags import FeatureFlags
from darwin.types import Goal
from darwin.worlds import AdaptiveRoomWorld


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


def _make_runtime() -> DarwinRuntime:
    world = AdaptiveRoomWorld(seed=1)
    adapter = RoomSimulationAdapter(world)
    darwin = Darwin(
        actions=ensure_chat_action(adapter.possible_actions()),
        seed=1, exploration_rate=0.0,
    )
    return DarwinRuntime(
        darwin=darwin, adapter=adapter,
        goal=Goal(desired={"room_bright": True}),
        interval=100.0,
    )


def test_default_runtime_uses_python_backends(clean_env) -> None:
    runtime = _make_runtime()
    assert runtime.feature_flags is not None
    assert runtime.feature_flags.mesh_backend == "python"
    assert runtime._torch_propagator is None


def test_torch_flag_routes_through(clean_env) -> None:
    os.environ["DARWIN_MESH_BACKEND"] = "torch"
    runtime = _make_runtime()
    from darwin.scale.torch_backend import torch_available
    if torch_available():
        assert runtime._torch_propagator is not None
    else:
        assert runtime._torch_propagator is None


def test_multiprocess_flag_collects_specs(clean_env) -> None:
    os.environ["DARWIN_MULTIPROCESS"] = "1"
    runtime = _make_runtime()
    if runtime.agent_registry is not None:
        assert len(runtime._agent_specs) == 6
