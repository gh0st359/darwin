"""VectorStore backends + sharded persistence."""

from __future__ import annotations

import pytest

from darwin.neural.vector_store import VectorStore, _HAS_NUMPY, cosine


def test_python_backend_seeded_init_is_deterministic():
    a = VectorStore(dim=16, backend="python")
    b = VectorStore(dim=16, backend="python")
    assert a.get("alpha") == b.get("alpha")
    assert a.get("beta") == b.get("beta")


def test_get_creates_lazily_and_tracks_order():
    s = VectorStore(dim=8, backend="python")
    assert s.size() == 0
    s.get("first")
    s.get("second")
    assert s.tokens() == ["first", "second"]
    assert s.size() == 2


def test_set_overrides_existing_vector():
    s = VectorStore(dim=4, backend="python")
    s.get("x")
    s.set("x", [1.0, 1.0, 1.0, 1.0])
    assert s.get("x") == [1.0, 1.0, 1.0, 1.0]


def test_nearest_excludes_self():
    s = VectorStore(dim=4, backend="python")
    s.set("a", [1.0, 0.0, 0.0, 0.0])
    s.set("b", [1.0, 0.1, 0.0, 0.0])
    s.set("c", [0.0, 1.0, 0.0, 0.0])
    near = s.nearest("a", k=2)
    assert near[0][0] == "b"
    assert near[0][1] > near[1][1]


def test_nearest_prefix_filter():
    s = VectorStore(dim=4, backend="python")
    s.set("act:x", [1.0, 0.0, 0.0, 0.0])
    s.set("pre:x", [1.0, 0.0, 0.0, 0.0])
    s.set("post:y", [0.5, 0.5, 0.0, 0.0])
    near = s.nearest("act:x", k=5, prefix="pre:")
    assert all(tok.startswith("pre:") for tok, _ in near)


def test_shard_round_trip(tmp_path):
    s = VectorStore(dim=8, backend="python")
    for tok in ("alpha", "beta", "gamma", "delta"):
        s.get(tok)
    shards = s.shard_to_disk(tmp_path)
    assert shards
    s2 = VectorStore(dim=8, backend="python")
    loaded = s2.load_shards(tmp_path)
    assert loaded == 4
    # Shards are float32 on disk — round-trip is exact within float32 precision.
    for tok in ("alpha", "beta", "gamma", "delta"):
        for x, y in zip(s2.get(tok), s.get(tok)):
            assert abs(x - y) < 1e-6


def test_shard_splits_at_byte_limit(tmp_path):
    # Force tiny shard limit so the splitter is exercised.
    s = VectorStore(dim=4, backend="python", shard_byte_limit=64)
    for i in range(20):
        s.get(f"token_{i}")
    shards = s.shard_to_disk(tmp_path)
    assert len(shards) > 1
    s2 = VectorStore(dim=4, backend="python", shard_byte_limit=64)
    s2.load_shards(tmp_path)
    assert s2.size() == 20


@pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
def test_numpy_backend_init_matches_python_seed():
    a = VectorStore(dim=16, backend="python")
    b = VectorStore(dim=16, backend="numpy")
    # Same hash-seed init means same starting vectors.
    va = a.get("hello")
    vb = b.get("hello")
    for x, y in zip(va, vb):
        assert abs(x - y) < 1e-6


@pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
def test_numpy_nearest_matches_python_ranking():
    pa = VectorStore(dim=8, backend="python")
    pb = VectorStore(dim=8, backend="numpy")
    for tok in ("apple", "banana", "carrot", "dog", "egg"):
        pa.get(tok)
        pb.get(tok)
    near_a = [tok for tok, _ in pa.nearest("apple", k=3)]
    near_b = [tok for tok, _ in pb.nearest("apple", k=3)]
    assert near_a == near_b


def test_shard_dim_mismatch_raises(tmp_path):
    s = VectorStore(dim=8, backend="python")
    s.get("x")
    s.shard_to_disk(tmp_path)
    s2 = VectorStore(dim=4, backend="python")
    with pytest.raises(ValueError):
        s2.load_shards(tmp_path)


def test_cosine_is_symmetric_and_one_for_self():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [4.0, 3.0, 2.0, 1.0]
    assert abs(cosine(a, b) - cosine(b, a)) < 1e-9
    assert abs(cosine(a, a) - 1.0) < 1e-9
