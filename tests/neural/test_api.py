"""TrainingClient — programmatic operator-facing API."""

from __future__ import annotations

from pathlib import Path

import pytest

from darwin.neural import TrainingClient


def test_headless_client_constructs_substrate(tmp_path: Path):
    client = TrainingClient(root=tmp_path)
    assert client.space is not None
    assert client.trainer is not None
    assert client.stream is not None
    assert client.tokenizer is not None


def test_feed_corpus_grows_vocab(tmp_path: Path):
    client = TrainingClient(root=tmp_path)
    client.feed_corpus("Darwin learns from operator-provided text.")
    assert client.flush(timeout=2.0)
    assert client.vocab() >= 4


def test_feed_corpus_iter_processes_each_chunk(tmp_path: Path):
    client = TrainingClient(root=tmp_path)
    client.feed_corpus_iter([
        "alpha beta gamma",
        "beta gamma delta",
        "gamma delta epsilon",
    ])
    assert client.flush(timeout=2.0)
    assert client.vocab() >= 5


def test_probe_returns_diagnostics(tmp_path: Path):
    client = TrainingClient(root=tmp_path)
    client.feed_corpus("alpha beta gamma alpha beta gamma")
    client.flush(timeout=2.0)
    report = client.probe("alpha")
    assert report["token"] == "alpha"
    assert report["in_vocab"] is True
    assert report["frequency"] >= 1
    assert isinstance(report["nearest"], list)


def test_probe_unknown_token_still_returns_record(tmp_path: Path):
    client = TrainingClient(root=tmp_path)
    out = client.probe("never_seen_token")
    assert out["token"] == "never_seen_token"
    # Either in_vocab=False, or the embed call materialised it as a fresh seed.
    assert "in_vocab" in out


def test_save_and_load_round_trip(tmp_path: Path):
    client = TrainingClient(root=tmp_path)
    client.feed_corpus("apple banana cherry apple banana cherry")
    client.flush(timeout=2.0)
    pre_vocab = client.vocab()
    client.save(label="v1")

    fresh = TrainingClient(root=tmp_path)
    fresh.load()
    assert fresh.vocab() == pre_vocab


def test_checkpoint_and_list(tmp_path: Path):
    client = TrainingClient(root=tmp_path)
    client.feed_corpus("one two three")
    client.flush(timeout=2.0)
    client.checkpoint("snap-a")
    assert "snap-a" in client.list_checkpoints()
    client.checkpoint("snap-b")
    assert set(client.list_checkpoints()) >= {"snap-a", "snap-b"}


def test_rollback_restores_prior_checkpoint(tmp_path: Path):
    client = TrainingClient(root=tmp_path)
    client.feed_corpus("alpha beta gamma")
    client.flush(timeout=2.0)
    client.checkpoint("baseline")
    # Capture the disk-canonical hash by reloading the just-saved state into
    # the live space — the save/load cycle quantises float64 → float32, so the
    # hash we want to compare against is the post-load one, not the in-memory
    # pre-save one.
    client.load()
    pre_hash = client.space.checkpoint_hash()
    pre_vocab = client.vocab()
    for _ in range(5):
        client.feed_corpus("delta epsilon zeta theta")
    client.flush(timeout=2.0)
    assert client.vocab() > pre_vocab
    client.rollback("baseline")
    assert client.space.checkpoint_hash() == pre_hash
    assert client.vocab() == pre_vocab


def test_cursor_persists_across_clients(tmp_path: Path):
    client = TrainingClient(root=tmp_path)
    client.feed_corpus("anything", source="my-source")
    assert client.cursor()["sources"]["my-source"]["bytes"] >= 8

    fresh = TrainingClient(root=tmp_path)
    cur = fresh.cursor()
    assert "my-source" in cur.get("sources", {})


def test_stats_includes_queue_and_checkpoint_info(tmp_path: Path):
    client = TrainingClient(root=tmp_path)
    client.feed_corpus("hello world")
    client.flush(timeout=2.0)
    stats = client.stats()
    for key in (
        "vocab_size", "loss_ewma", "queue_size", "stream_chunks_seen",
        "stream_pairs_submitted", "checkpoints",
    ):
        assert key in stats


def test_runtime_bound_client_shares_space_with_runtime():
    """When a runtime is provided, the client binds to its live substrate."""
    from darwin.agent import Darwin
    from darwin.embodiment import RoomSimulationAdapter
    from darwin.runtime import DarwinRuntime, ensure_chat_action
    from darwin.types import Goal
    from darwin.worlds import AdaptiveRoomWorld

    world = AdaptiveRoomWorld(seed=11)
    adapter = RoomSimulationAdapter(world)
    darwin = Darwin(
        actions=ensure_chat_action(adapter.possible_actions()),
        seed=11, exploration_rate=0.0,
    )
    runtime = DarwinRuntime(
        darwin=darwin, adapter=adapter,
        goal=Goal(desired={"room_bright": True}),
        interval=100.0,
    )
    client = TrainingClient(runtime=runtime)
    assert client.space is runtime.embedding_space
    assert client.trainer is runtime.embedding_trainer
    assert client.stream is runtime.corpus_stream
