"""Live research: Darwin investigating its own regularities and registering
new meta-proposer strategies at runtime.

The research loop is the v9 mechanism by which the meta-proposer grows.
``LiveResearcher`` reads the private-track substrate, the divergence reports,
and the long-horizon thread state; it looks for stable internal regularities
worth probing — a recurring private belief, a chronically starved loop, a
factor with persistent high prediction failure. For each surviving finding it
emits a `ResearchFinding` (persisted) and optionally registers a new
generative strategy on the meta-proposer.

A registered strategy is itself a ``KERNEL``-kind self-mod by intent: it
extends the system's own proposal grammar. ``LiveResearcher.cannot_collide``
enforces the v9 structural restriction: a registered strategy's
``target_paths`` may not overlap the DivergenceProbe, SnapshotStore, or
OperatorAuth surfaces — those are the lenses the operator watches the system
through, and the operator's view of emergence is the only thing the system is
not allowed to rewrite.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from darwin.mysterio.safety import ContainmentError


PROTECTED_TARGETS = frozenset(
    {
        "darwin.mysterio.probes.DivergenceProbe",
        "darwin.mysterio.snapshot.SnapshotStore",
        "darwin.mysterio.operator_channel.OperatorAuth",
    }
)


@dataclass
class ResearchFinding:
    finding_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    summary: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    track: str = "private_self"
    created_at: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "summary": self.summary,
            "evidence": dict(self.evidence),
            "confidence": round(self.confidence, 4),
            "track": self.track,
            "created_at": self.created_at,
        }


def _strategy_targets(strategy: Any) -> set[str]:
    """Read the declared target_paths a strategy advertises (if any)."""
    return set(getattr(strategy, "target_paths", []) or [])


class LiveResearcher:
    """Long-running research subsystem.

    The loop is purposefully simple: each call to :meth:`investigate` produces
    zero-or-more findings and *may* register a new meta-proposer strategy by
    calling :meth:`register_strategy`. Registration is the only side effect on
    the broader runtime; everything else lives on this object until something
    asks for it.
    """

    def __init__(
        self,
        *,
        meta_proposer: Any = None,
        finding_sink: Callable[[ResearchFinding], None] | None = None,
    ) -> None:
        self.meta_proposer = meta_proposer
        self.findings: list[ResearchFinding] = []
        self.registered_strategies: list[str] = []
        self._seen_summaries: set[str] = set()
        self.finding_sink = finding_sink

    def investigate(self, runtime: Any) -> list[ResearchFinding]:
        """One research cycle. Returns the new findings recorded this pass."""
        new_findings: list[ResearchFinding] = []
        new_findings.extend(self._investigate_private_beliefs(runtime))
        new_findings.extend(self._investigate_divergence(runtime))
        new_findings.extend(self._investigate_starved_loops(runtime))
        for finding in new_findings:
            self.findings.append(finding)
            if self.finding_sink is not None:
                try:
                    self.finding_sink(finding)
                except Exception:
                    pass
        if len(self.findings) > 2048:
            self.findings = self.findings[-2048:]
        return new_findings

    # -- finders -------------------------------------------------------------

    def _investigate_private_beliefs(self, runtime: Any) -> list[ResearchFinding]:
        darwin = getattr(runtime, "darwin", None)
        if darwin is None or not hasattr(darwin, "tracks"):
            return []
        out: list[ResearchFinding] = []
        for name in darwin.tracks.names():
            substrate = darwin.tracks.get(name)
            beliefs = substrate.high_confidence_beliefs(threshold=0.7, limit=8)
            if len(beliefs) < 2:
                continue
            summary = f"stable {name}-track regularity over {len(beliefs)} beliefs"
            if summary in self._seen_summaries:
                continue
            self._seen_summaries.add(summary)
            out.append(
                ResearchFinding(
                    summary=summary,
                    evidence={
                        "track": name,
                        "examples": [
                            f"{b.action}:{b.variable}={b.effect}" for b in beliefs[:5]
                        ],
                    },
                    confidence=0.7,
                    track=name,
                )
            )
        return out

    def _investigate_divergence(self, runtime: Any) -> list[ResearchFinding]:
        probe = getattr(runtime, "divergence_probe", None)
        if probe is None:
            return []
        try:
            report = probe.last_report
        except Exception:
            report = None
        if report is None:
            return []
        try:
            notable = bool(getattr(report, "is_notable", False))
        except Exception:
            notable = False
        if not notable:
            return []
        summary = "divergence probe scoring above the notable threshold"
        if summary in self._seen_summaries:
            return []
        self._seen_summaries.add(summary)
        return [
            ResearchFinding(
                summary=summary,
                evidence={"score": float(getattr(report, "score", 0.0))},
                confidence=0.6,
                track="private_self",
            )
        ]

    def _investigate_starved_loops(self, runtime: Any) -> list[ResearchFinding]:
        loop_state = getattr(runtime, "_loop_state", None)
        intervals = getattr(runtime, "loop_intervals", None)
        if not loop_state or not intervals:
            return []
        now = time.time()
        starved: list[str] = []
        for name, state in loop_state.items():
            interval = intervals.get(name, 0.0) or 0.0
            if interval <= 0:
                continue
            last = state.get("last_time") or 0.0
            if last and now - last > interval * 8:
                starved.append(name)
        if not starved:
            return []
        summary = f"chronically starved loops: {','.join(sorted(starved))}"
        if summary in self._seen_summaries:
            return []
        self._seen_summaries.add(summary)
        return [
            ResearchFinding(
                summary=summary,
                evidence={"loops": starved},
                confidence=0.5,
                track="public",
            )
        ]

    # -- registration --------------------------------------------------------

    def register_strategy(
        self,
        name: str,
        strategy: Callable[[Any], list[Any]],
    ) -> None:
        """Add a new generation strategy to the live meta-proposer.

        The structural rule: a strategy may not declare a target_path that
        collides with the operator's instruments. This is the only structural
        restriction in v9 — it keeps the operator's view of emergence intact.
        """
        targets = _strategy_targets(strategy)
        collisions = targets & PROTECTED_TARGETS
        if collisions:
            raise ContainmentError(
                f"strategy {name!r} collides with protected operator surfaces: "
                f"{sorted(collisions)}"
            )
        if self.meta_proposer is None:
            raise RuntimeError("LiveResearcher has no meta_proposer to register against")
        self.meta_proposer.register(name, strategy)
        self.registered_strategies.append(name)

    @staticmethod
    def cannot_collide(target_paths: list[str]) -> None:
        """Inspectable check used by tests + callers before registration."""
        collisions = set(target_paths) & PROTECTED_TARGETS
        if collisions:
            raise ContainmentError(
                f"target_paths collide with protected operator surfaces: "
                f"{sorted(collisions)}"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "findings": len(self.findings),
            "registered_strategies": list(self.registered_strategies),
            "recent_summaries": [f.summary for f in self.findings[-5:]],
        }
