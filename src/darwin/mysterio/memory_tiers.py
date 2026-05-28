"""Five-tier memory: episodic → semantic → conceptual → archetypal → narrative.

Single-tier memory says everything in one table. That works at hundreds of
transitions. At GB scale across months it stops working: a system that
remembers everything equally remembers nothing well, and a planner that
re-reads every transition every step grinds. Multi-tier memory is the
tractability story.

Each tier consolidates from the one below it at its own cadence:

  EpisodicMemory      seconds → minutes  — raw transitions
  SemanticMemory      minutes → hours    — facts, relations, generalizations
  ConceptualMemory    hours              — concept clusters, abstractions
  ArchetypalMemory    days               — recurring patterns, prototypes
  NarrativeMemory     weeks              — long-arc autobiographical chunks

The shape implements the cadence: every tier has a ``consolidate()`` method
that pulls salient material from the tier below and a ``cadence_seconds``
that says how often it runs. The kernel drives them via the supervisor;
nothing here knows about threads.

Each tier is also keyed by track. A v7-private rollout's consolidation lives
on the private side and never escapes to the public tiers.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from darwin.mysterio.tracks import PUBLIC_TRACK


@dataclass
class MemoryItem:
    text: str
    track: str = PUBLIC_TRACK
    salience: float = 0.0
    tier: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "track": self.track,
            "salience": round(self.salience, 4),
            "tier": self.tier,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }


class _Tier:
    """Common bounded-deque store with track-aware reads."""

    tier_name: str = "tier"
    cadence_seconds: float = 60.0
    capacity: int = 8192

    def __init__(self) -> None:
        self._items: deque[MemoryItem] = deque(maxlen=self.capacity)

    def add(self, item: MemoryItem) -> None:
        item.tier = self.tier_name
        self._items.append(item)

    def by_track(self, track: str = PUBLIC_TRACK, limit: int = 64) -> list[MemoryItem]:
        return [m for m in self._items if m.track == track][-limit:]

    def all(self) -> list[MemoryItem]:
        return list(self._items)

    def size(self) -> int:
        return len(self._items)

    def consolidate(self, lower: "_Tier | None") -> dict[str, Any]:
        return {"tier": self.tier_name, "consolidated": 0}


class EpisodicMemoryTier(_Tier):
    tier_name = "episodic"
    cadence_seconds = 5.0
    capacity = 65536

    def ingest(self, transition: Any, track: str = PUBLIC_TRACK) -> None:
        text = f"{getattr(transition, 'action', '?')}: " + ", ".join(
            f"{k}={v}" for k, v in dict(getattr(transition, "after", {})).items()
        )
        salience = abs(float(getattr(transition, "reward", 0.0)))
        self.add(MemoryItem(text=text, track=track, salience=salience))


class SemanticMemoryTier(_Tier):
    """Consolidates episodes into compressed facts ('flip_switch ⇒ room_bright')."""

    tier_name = "semantic"
    cadence_seconds = 60.0
    capacity = 16384

    def consolidate(self, lower: _Tier | None) -> dict[str, Any]:
        if lower is None:
            return {"tier": self.tier_name, "consolidated": 0}
        counter: Counter[tuple[str, str]] = Counter()
        track_of: dict[tuple[str, str], str] = {}
        for item in lower.all():
            head = item.text.split(":", 1)[0].strip() or item.text[:24]
            tail = (item.text.split(":", 1)[1] if ":" in item.text else "").strip()
            for piece in tail.split(","):
                piece = piece.strip()
                if not piece:
                    continue
                key = (head, piece)
                counter[key] += 1
                track_of[key] = item.track
        promoted = 0
        for (head, piece), count in counter.most_common(64):
            if count < 2:
                continue
            self.add(
                MemoryItem(
                    text=f"{head} ⇒ {piece}",
                    track=track_of.get((head, piece), PUBLIC_TRACK),
                    salience=float(count),
                    metadata={"support": count},
                )
            )
            promoted += 1
        return {"tier": self.tier_name, "consolidated": promoted}


class ConceptualMemoryTier(_Tier):
    """Clusters semantic facts that share a head into a concept."""

    tier_name = "conceptual"
    cadence_seconds = 600.0
    capacity = 4096

    def consolidate(self, lower: _Tier | None) -> dict[str, Any]:
        if lower is None:
            return {"tier": self.tier_name, "consolidated": 0}
        heads: dict[str, list[MemoryItem]] = {}
        for item in lower.all():
            head = item.text.split("⇒", 1)[0].strip()
            heads.setdefault(head, []).append(item)
        promoted = 0
        for head, group in heads.items():
            if len(group) < 3:
                continue
            track = max(
                (g.track for g in group),
                key=lambda t: sum(1 for g in group if g.track == t),
            )
            self.add(
                MemoryItem(
                    text=f"concept[{head}] supports {len(group)} relations",
                    track=track,
                    salience=float(len(group)),
                    metadata={"members": [g.text for g in group[:8]]},
                )
            )
            promoted += 1
        return {"tier": self.tier_name, "consolidated": promoted}


class ArchetypalMemoryTier(_Tier):
    """Prototypes: long-stable concepts that have been re-promoted repeatedly."""

    tier_name = "archetypal"
    cadence_seconds = 3600.0 * 12
    capacity = 1024

    def consolidate(self, lower: _Tier | None) -> dict[str, Any]:
        if lower is None:
            return {"tier": self.tier_name, "consolidated": 0}
        ranked = sorted(lower.all(), key=lambda i: i.salience, reverse=True)
        existing = {i.text for i in self._items}
        promoted = 0
        for item in ranked[:8]:
            arch_text = f"archetype: {item.text}"
            if arch_text in existing:
                continue
            self.add(
                MemoryItem(
                    text=arch_text,
                    track=item.track,
                    salience=item.salience * 1.5,
                    metadata=dict(item.metadata),
                )
            )
            promoted += 1
        return {"tier": self.tier_name, "consolidated": promoted}


class NarrativeMemoryTier(_Tier):
    """Long-arc autobiographical chunks; consolidates from a narrative thread."""

    tier_name = "narrative"
    cadence_seconds = 3600.0 * 24 * 7
    capacity = 4096

    def consolidate_narrative(self, chunks: Iterable[Any]) -> dict[str, Any]:
        promoted = 0
        for chunk in chunks:
            self.add(
                MemoryItem(
                    text=getattr(chunk, "text", ""),
                    track="private_self",
                    salience=float(len(getattr(chunk, "text", "").split())),
                    metadata={"chunk_id": getattr(chunk, "chunk_id", "")},
                )
            )
            promoted += 1
        return {"tier": self.tier_name, "consolidated": promoted}


class MemoryTierStack:
    """The five tiers wired in the correct consolidation order."""

    def __init__(self) -> None:
        self.episodic = EpisodicMemoryTier()
        self.semantic = SemanticMemoryTier()
        self.conceptual = ConceptualMemoryTier()
        self.archetypal = ArchetypalMemoryTier()
        self.narrative = NarrativeMemoryTier()

    def ingest_transition(self, transition: Any, track: str = PUBLIC_TRACK) -> None:
        self.episodic.ingest(transition, track=track)

    def step(self) -> dict[str, Any]:
        """Run one consolidation pass through the stack."""
        sem = self.semantic.consolidate(self.episodic)
        con = self.conceptual.consolidate(self.semantic)
        arc = self.archetypal.consolidate(self.conceptual)
        return {"semantic": sem, "conceptual": con, "archetypal": arc}

    def summary(self) -> dict[str, Any]:
        return {
            "episodic": self.episodic.size(),
            "semantic": self.semantic.size(),
            "conceptual": self.conceptual.size(),
            "archetypal": self.archetypal.size(),
            "narrative": self.narrative.size(),
        }
