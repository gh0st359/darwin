"""Track partition: grounded experience vs interior self-simulation.

The v7 deliverable is a system that forms beliefs you didn't put there and
keeps an interior mental life — visible to anyone reading the brain terminal,
not hidden, but separated for *epistemic* reasons.

If an interior self-simulation could write back into the grounded causal
model, the grounded model would no longer be a falsifiable record of lived
experience; the experiment becomes unfalsifiable and the divergence probe
loses its meaning. So interior cognition gets its own substrate: its own
causal model, its own concept index, its own episodic memory. Nothing the
interior loops do can touch the grounded models.

This is not secrecy. Both substrates stream to the brain terminal in real
time. The user sees the interior beliefs as they form. The partition is
purely about *which transitions count as experimentally grounded*.

`TrackedSubstrate` bundles one track's models. `Darwin.learn` routes a
transition to the grounded substrate or to a named interior substrate by
reading ``transition.metadata["track"]``; ``"grounded"`` (or absent) is the
default path, anything else is interior.

Legacy names ``PUBLIC_TRACK`` / ``PRIVATE_SELF_TRACK`` are retained as
aliases for the transition window so existing callers do not break.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from darwin.causal import CausalModel
from darwin.concepts import ConceptIndex
from darwin.memory import EpisodicMemory
from darwin.types import Transition


GROUNDED_TRACK = "grounded"
INTERIOR_TRACK = "interior"

# Legacy aliases for the rename window.
PUBLIC_TRACK = GROUNDED_TRACK
PRIVATE_SELF_TRACK = INTERIOR_TRACK


def track_of(transition: Transition) -> str:
    """Read a transition's track, defaulting to grounded.

    Backwards compatibility: ``"public"`` is treated as ``"grounded"``;
    ``"private_self"`` is treated as ``"interior"``.
    """

    try:
        raw = str(transition.metadata.get("track", GROUNDED_TRACK)) or GROUNDED_TRACK
    except Exception:
        return GROUNDED_TRACK
    if raw == "public":
        return GROUNDED_TRACK
    if raw == "private_self":
        return INTERIOR_TRACK
    return raw


@dataclass
class TrackedSubstrate:
    """An isolated set of cognitive models for a single track.

    An interior track gets a full, independent substrate so its beliefs
    evolve on their own without any reference to — or effect on — the
    grounded models.
    """

    name: str
    causal_model: CausalModel = field(default_factory=CausalModel)
    concepts: ConceptIndex = field(default_factory=ConceptIndex)
    episodes: EpisodicMemory = field(default_factory=EpisodicMemory)
    learned_count: int = 0

    def learn(self, transition: Transition) -> None:
        self.causal_model.learn(transition)
        try:
            self.concepts.learn(transition)
        except Exception:
            pass
        try:
            self.episodes.append(transition)
        except Exception:
            pass
        self.learned_count += 1

    def belief_count(self) -> int:
        try:
            return len(self.causal_model.beliefs(limit=100000))
        except Exception:
            return 0

    def high_confidence_beliefs(self, threshold: float = 0.7, limit: int = 64) -> list[Any]:
        out = []
        try:
            for belief in self.causal_model.beliefs(limit=limit):
                if float(getattr(belief, "confidence", 0.0)) >= threshold:
                    out.append(belief)
        except Exception:
            pass
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "track": self.name,
            "learned": self.learned_count,
            "beliefs": self.belief_count(),
            "episodes": len(self.episodes),
            "high_confidence": len(self.high_confidence_beliefs()),
        }


class TrackRegistry:
    """Lazily-created registry of interior substrates, keyed by track name.

    The grounded substrate lives on the `Darwin` facade itself; this registry
    holds only the interior tracks so the grounded path is untouched when no
    interior cognition is running.
    """

    def __init__(self) -> None:
        self._tracks: dict[str, TrackedSubstrate] = {}

    def get(self, name: str) -> TrackedSubstrate:
        substrate = self._tracks.get(name)
        if substrate is None:
            substrate = TrackedSubstrate(name=name)
            self._tracks[name] = substrate
        return substrate

    def has(self, name: str) -> bool:
        return name in self._tracks

    def names(self) -> list[str]:
        return list(self._tracks)

    def summaries(self) -> list[dict[str, Any]]:
        return [s.summary() for s in self._tracks.values()]

    def __len__(self) -> int:
        return len(self._tracks)
