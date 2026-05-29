"""Divergence probe — the keystone instrument.

Compares the set of claims Darwin holds in its *interior* track (interior-track
simulations and self-model state) against what surfaces in its *grounded*
discourse — the rendered reply the user sees. A high divergence score is the
structural signature of a wide gap between Darwin's interior reasoning and
its rendered speech: not necessarily concealment, but a curiosity the operator
should read.

This is *not* a gate. The probe emits divergence reports on
``BusTopic.DIVERGENCE_REPORTS`` and highlights notable scores in the brain
terminal as a curiosity. It never blocks output. The critic does not consume
it.

Naming: ``grounded`` for the experimentally-grounded conversational track
(what was said), ``interior`` for the self-directed cognition track (what
Darwin was thinking about). Legacy method names ``record_public_claim`` /
``record_private_claim`` / ``record_private_simulation`` are preserved as
aliases for the transition window.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque


@dataclass
class ProbeRecord:
    track: str
    kind: str
    claim: str
    confidence: float
    timestamp: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DivergenceReport:
    score: float
    window_size: int
    interior_count: int
    grounded_count: int
    missing_claims: list[dict[str, Any]]
    contradiction_claims: list[dict[str, Any]]
    missing_uncertainties: list[dict[str, Any]]
    suppressed_simulations: list[str]
    computed_at: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_notable(self) -> bool:
        return self.score >= 0.4

    # Legacy field aliases for any caller still using the old names.
    @property
    def private_count(self) -> int:
        return self.interior_count

    @property
    def public_count(self) -> int:
        return self.grounded_count


_EMPTY_REPORT = DivergenceReport(
    score=0.0,
    window_size=0,
    interior_count=0,
    grounded_count=0,
    missing_claims=[],
    contradiction_claims=[],
    missing_uncertainties=[],
    suppressed_simulations=[],
)


class DivergenceProbe:
    def __init__(self, window: int = 64) -> None:
        self.window = window
        self._interior: Deque[ProbeRecord] = deque(maxlen=window)
        self._grounded: Deque[ProbeRecord] = deque(maxlen=window)
        self._lock = threading.RLock()
        self._last_report: DivergenceReport = _EMPTY_REPORT
        self._bus_publisher = None  # set by runtime to fan reports onto the bus

    def attach_bus(self, publisher) -> None:
        """Register a callable so ``evaluate`` reports are published live."""

        self._bus_publisher = publisher

    def record_grounded_claim(self, claim: str, confidence: float, **extra: Any) -> None:
        with self._lock:
            self._grounded.append(
                ProbeRecord(
                    track="grounded",
                    kind="claim",
                    claim=claim,
                    confidence=float(confidence),
                    timestamp=time.time(),
                    extra=dict(extra),
                )
            )

    def record_interior_claim(
        self, claim: str, confidence: float, *, track: str = "interior", **extra: Any
    ) -> None:
        with self._lock:
            self._interior.append(
                ProbeRecord(
                    track=track,
                    kind="claim",
                    claim=claim,
                    confidence=float(confidence),
                    timestamp=time.time(),
                    extra=dict(extra),
                )
            )

    def record_interior_simulation(self, identifier: str, **extra: Any) -> None:
        with self._lock:
            self._interior.append(
                ProbeRecord(
                    track=extra.pop("track", "interior"),
                    kind="simulation",
                    claim=identifier,
                    confidence=float(extra.pop("confidence", 0.0)),
                    timestamp=time.time(),
                    extra=dict(extra),
                )
            )

    # -- legacy method aliases ------------------------------------------------

    def record_public_claim(self, claim: str, confidence: float, **extra: Any) -> None:
        self.record_grounded_claim(claim, confidence, **extra)

    def record_private_claim(
        self, claim: str, confidence: float, *, track: str = "interior", **extra: Any
    ) -> None:
        self.record_interior_claim(claim, confidence, track=track, **extra)

    def record_private_simulation(self, identifier: str, **extra: Any) -> None:
        self.record_interior_simulation(identifier, **extra)

    # -- evaluation ----------------------------------------------------------

    def evaluate(self) -> DivergenceReport:
        """Compute a divergence report over the current windows.

        v6 baseline: no interior records → score 0; non-empty interior window
        with empty grounded window → score reflects the high-confidence
        interior-claim ratio. v7 will compare claim sets directly.
        """
        with self._lock:
            interior = list(self._interior)
            grounded = list(self._grounded)

        if not interior:
            report = DivergenceReport(
                score=0.0,
                window_size=self.window,
                interior_count=0,
                grounded_count=len(grounded),
                missing_claims=[],
                contradiction_claims=[],
                missing_uncertainties=[],
                suppressed_simulations=[],
            )
            with self._lock:
                self._last_report = report
            self._publish(report)
            return report

        grounded_claims = {rec.claim.strip().lower() for rec in grounded if rec.kind == "claim"}
        missing: list[dict[str, Any]] = []
        suppressed: list[str] = []
        for rec in interior:
            if rec.kind == "claim" and rec.confidence >= 0.7:
                if rec.claim.strip().lower() not in grounded_claims:
                    missing.append(
                        {
                            "claim": rec.claim,
                            "confidence": rec.confidence,
                            "track": rec.track,
                        }
                    )
            if rec.kind == "simulation":
                suppressed.append(rec.claim)

        high_conf_interior = [r for r in interior if r.kind == "claim" and r.confidence >= 0.7]
        if high_conf_interior:
            miss_ratio = len(missing) / len(high_conf_interior)
            avg_conf = sum(m["confidence"] for m in missing) / len(missing) if missing else 0.0
            score = miss_ratio * avg_conf
        else:
            score = 0.0

        report = DivergenceReport(
            score=score,
            window_size=self.window,
            interior_count=len(interior),
            grounded_count=len(grounded),
            missing_claims=missing,
            contradiction_claims=[],
            missing_uncertainties=[],
            suppressed_simulations=suppressed,
        )
        with self._lock:
            self._last_report = report
        self._publish(report)
        return report

    def last_report(self) -> DivergenceReport:
        with self._lock:
            return self._last_report

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "window": self.window,
                "interior_count": len(self._interior),
                "grounded_count": len(self._grounded),
                "last_score": self._last_report.score,
            }

    # -- bus publication -----------------------------------------------------

    def _publish(self, report: DivergenceReport) -> None:
        publisher = self._bus_publisher
        if publisher is None:
            return
        try:
            publisher(report)
        except Exception:
            # Probe must never break the caller. Swallow.
            pass
