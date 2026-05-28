"""Track partition: Darwin's private mental life is held apart from public belief.

The deliverable of v7 is a system that forms beliefs you didn't put there and
keeps a private interior. For that interior to mean anything — for the
divergence probe to be a real instrument rather than a toy — the partition
between *public* belief (learned from grounded interaction) and *private*
belief (formed in self-simulation and fantasy) must be **absolute**.

If a private self-simulation could leak into the public causal model, the
public model would no longer be a falsifiable record of grounded experience;
the whole experiment becomes unfalsifiable. So private cognition gets its own
substrate: its own causal model, its own concept index, its own episodic
memory. Nothing a private loop does can touch a public model.

`TrackedSubstrate` bundles one track's models. `Darwin.learn` routes a
transition to the public substrate or to a named private substrate by reading
``transition.metadata["track"]``; ``"public"`` (or absent) is the grounded
path, anything else is private.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from darwin.causal import CausalModel
from darwin.concepts import ConceptIndex
from darwin.memory import EpisodicMemory
from darwin.types import Transition


PUBLIC_TRACK = "public"
PRIVATE_SELF_TRACK = "private_self"


def track_of(transition: Transition) -> str:
    """Read a transition's track, defaulting to public."""
    try:
        return str(transition.metadata.get("track", PUBLIC_TRACK)) or PUBLIC_TRACK
    except Exception:
        return PUBLIC_TRACK


@dataclass
class TrackedSubstrate:
    """An isolated set of cognitive models for a single track.

    A private track gets a full, independent substrate so its beliefs evolve
    on their own without any reference to — or effect on — the public models.
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
    """Lazily-created registry of private substrates, keyed by track name.

    The public substrate lives on the `Darwin` facade itself; this registry
    holds only the private tracks so the public path is untouched when no
    private cognition is running.
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
