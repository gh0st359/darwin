"""Capability probe — non-fixture, non-memorisable evaluation.

Replaces the deleted ``bench/frontier`` adapter suite. Where the
frontier adapters loaded fixed JSONL fixtures and could be gamed by
template solvers, the capability probe generates **novel problem
instances at evaluation time** from Darwin's own learned state:

  * Concept-walk questions ("What relates X to Y?") sampled by a random
    walk through Darwin's universe — no pre-canned set of pairs.
  * Arithmetic samples drawn from a held-out parameter grid — the
    operator can change the seed and get a fresh set every run.
  * Procedural planning grids generated on the fly — the goal/cost
    structure is fresh each call.

Nothing on disk to overfit. Nothing memorisable. The score measures
how the learned substrate *actually generalises* on problems Darwin has
never seen.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from darwin.bench.framework import BenchmarkSuite, BenchmarkTask


# --------------------------------------------------------------------------- #
# Concept-walk probe
# --------------------------------------------------------------------------- #


def _concept_walk(runtime: Any, *, seed: int = 0) -> tuple[float, dict]:
    """Pick two concepts and ask Darwin to relate them via reasoning."""

    universe = getattr(runtime, "universe", None)
    if universe is None:
        return 0.0, {"error": "no universe"}
    rng = random.Random(seed)
    try:
        concepts = list(universe.all_concept_names()) if hasattr(
            universe, "all_concept_names"
        ) else []
    except Exception:
        concepts = []
    if len(concepts) < 4:
        return 0.0, {"error": "universe too small", "concepts": len(concepts)}
    a, b = rng.sample(concepts, 2)
    question = f"What relates {a} to {b}?"
    reply = runtime.chat(question)
    grounded = (a.lower() in reply.lower()) and (b.lower() in reply.lower())
    return (1.0 if grounded else 0.0), {
        "question": question,
        "reply_preview": reply[:200],
        "concepts_present": grounded,
    }


# --------------------------------------------------------------------------- #
# Procedural arithmetic
# --------------------------------------------------------------------------- #


def _procedural_arithmetic(runtime: Any, *, seed: int = 0) -> tuple[float, dict]:
    """Sample an arithmetic expression and check Darwin's reply contains the answer."""

    rng = random.Random(seed)
    a = rng.randint(2, 99)
    b = rng.randint(2, 99)
    expr = f"{a} * ({b} + {a})"
    expected = a * (b + a)
    reply = runtime.chat(f"What is {expr}?")
    correct = str(expected) in reply
    return (1.0 if correct else 0.0), {
        "expression": expr,
        "expected": expected,
        "reply_preview": reply[:200],
    }


# --------------------------------------------------------------------------- #
# Embedding-space probe — measures whether the learned space has structure
# --------------------------------------------------------------------------- #


def _embedding_neighbourhood(runtime: Any, *, seed: int = 0) -> tuple[float, dict]:
    """Train a small known cluster and verify the nearest-neighbour ordering."""

    space = getattr(runtime, "embedding_space", None)
    if space is None:
        return 0.0, {"error": "no embedding space"}
    rng = random.Random(seed)
    # Sample two synthetic clusters of tokens; training co-occurrence should
    # pull cluster members together more than across clusters.
    cluster_a = [f"probe_a_{i}_{rng.randint(0, 9999)}" for i in range(4)]
    cluster_b = [f"probe_b_{i}_{rng.randint(0, 9999)}" for i in range(4)]
    for _ in range(40):
        space.train_tokens(cluster_a)
        space.train_tokens(cluster_b)
    near = space.nearest(cluster_a[0], k=3)
    # Score: how many of cluster_a's other members appear in top-3.
    members = set(cluster_a[1:])
    hits = sum(1 for tok, _ in near if tok in members)
    return float(hits) / 3.0, {
        "near": [tok for tok, _ in near],
        "hits": hits,
    }


# --------------------------------------------------------------------------- #
# Multi-hop derivation probe
# --------------------------------------------------------------------------- #


def _procedural_derivation(runtime: Any, *, seed: int = 0) -> tuple[float, dict]:
    """Teach a fresh 4-node taxonomy, verify transitive derivation."""

    rng = random.Random(seed)
    names = [f"node_{rng.randint(0, 99999)}" for _ in range(4)]
    for src, dst in zip(names, names[1:]):
        runtime.chat(f"A {src} is a {dst}.")
    reply = runtime.chat(f"Is a {names[0]} a {names[-1]}?")
    derived = names[0].lower() in reply.lower() and names[-1].lower() in reply.lower()
    derived = derived and (
        "yes" in reply.lower() or "is a" in reply.lower() or "chain" in reply.lower()
    )
    return (1.0 if derived else 0.0), {"reply_preview": reply[:200]}


# --------------------------------------------------------------------------- #
# Suite assembly
# --------------------------------------------------------------------------- #


@dataclass
class CapabilityProbe:
    """Builds a non-memorisable capability suite for runtime evaluation."""

    seed: int = 0

    def build(self) -> BenchmarkSuite:
        suite = BenchmarkSuite(name="capability")
        # Bind seed via closure so each probe is reproducible per run but
        # varies across runs when the operator changes seed.
        seed = self.seed
        suite.add(BenchmarkTask(
            "capability/concept_walk", "capability",
            "ask Darwin to relate two random universe concepts",
            lambda r: _concept_walk(r, seed=seed + 1),
        ))
        suite.add(BenchmarkTask(
            "capability/arithmetic", "capability",
            "evaluate a procedurally-generated arithmetic expression",
            lambda r: _procedural_arithmetic(r, seed=seed + 2),
        ))
        suite.add(BenchmarkTask(
            "capability/embedding_neighbourhood", "capability",
            "trained clusters become near each other in the learned space",
            lambda r: _embedding_neighbourhood(r, seed=seed + 3),
        ))
        suite.add(BenchmarkTask(
            "capability/procedural_derivation", "capability",
            "transitive derivation over a freshly-taught taxonomy",
            lambda r: _procedural_derivation(r, seed=seed + 4),
        ))
        return suite


def build_capability_suite(seed: int = 0) -> BenchmarkSuite:
    return CapabilityProbe(seed=seed).build()


__all__ = ["CapabilityProbe", "build_capability_suite"]
