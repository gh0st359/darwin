from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


@dataclass
class TrainingPair:
    """A single (structured plan -> high-quality rendering) example.

    These pairs are what we will eventually use to fine-tune the DLM
    (gemma-3-270m) with LoRA. The plan side is fully self-supervised
    from Darwin's own structured output; the rendering side comes from
    the deterministic composer until a curated human pass replaces it.
    """

    plan_id: str
    user_text: str
    plan_payload: dict[str, Any]
    rendering: str
    renderer: str
    critique_passed: bool
    quality: float = 0.5
    accepted: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "user_text": self.user_text,
            "plan_payload": self.plan_payload,
            "rendering": self.rendering,
            "renderer": self.renderer,
            "critique_passed": self.critique_passed,
            "quality": self.quality,
            "accepted": self.accepted,
            "timestamp": self.timestamp,
        }


class TrainingDataCollector:
    """Collects (plan -> rendering) pairs to fine-tune the DLM.

    The collector is deliberately *additive* and never modifies Darwin's
    behaviour. Pairs whose critique passed are tagged as candidate
    training data; the rest are still kept for analysis but flagged.

    Phase 4 design intent (see docs/V2_ARCHITECTURE.md):
      * Primary corpus: deterministic-composer renderings of real plans
        from extensive live + simulation sessions. No external model is
        ever used to generate Darwin's *thinking*.
      * Curated pass: a small set of (plan -> rendering) pairs may be
        edited by hand to produce a high-quality fluency target.
      * Optional, *one-shot*, heavily filtered: a larger model may be
        used once to polish renderings into more natural prose, but
        only after the FaithfulnessValidator confirms each candidate.
        After that one-shot pass, the larger model is never used again.
    """

    def __init__(
        self,
        path: Path | str | None = None,
        enabled: bool = True,
    ) -> None:
        from darwin.paths import dlm_training_pairs_path

        self.path = Path(path) if path is not None else dlm_training_pairs_path()
        self.enabled = enabled
        self._lock = threading.RLock()
        self.pairs: list[TrainingPair] = []
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(
        self,
        plan_id: str,
        user_text: str,
        plan_payload: dict[str, Any],
        rendering: str,
        renderer: str,
        critique_passed: bool,
        quality: float | None = None,
    ) -> TrainingPair:
        pair = TrainingPair(
            plan_id=plan_id,
            user_text=user_text,
            plan_payload=plan_payload,
            rendering=rendering,
            renderer=renderer,
            critique_passed=critique_passed,
            quality=quality if quality is not None else (0.8 if critique_passed else 0.3),
            accepted=critique_passed,
        )
        self.pairs.append(pair)
        if not self.enabled:
            return pair
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(pair.to_record(), default=_json_default))
                handle.write("\n")
        return pair

    def export(
        self,
        destination: Path | str,
        min_quality: float = 0.7,
        renderer: str | None = None,
    ) -> int:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        accepted = self._eligible_pairs(min_quality=min_quality, renderer=renderer)
        with target.open("w", encoding="utf-8") as handle:
            for pair in accepted:
                handle.write(json.dumps(pair.to_record(), default=_json_default))
                handle.write("\n")
        return len(accepted)

    def load_existing(self) -> list[TrainingPair]:
        if not self.path.exists():
            return []
        records: list[TrainingPair] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                records.append(
                    TrainingPair(
                        plan_id=data["plan_id"],
                        user_text=data["user_text"],
                        plan_payload=data["plan_payload"],
                        rendering=data["rendering"],
                        renderer=data["renderer"],
                        critique_passed=data.get("critique_passed", True),
                        quality=data.get("quality", 0.5),
                        accepted=data.get("accepted", False),
                        timestamp=data.get("timestamp", time.time()),
                    )
                )
        self.pairs = records
        return records

    def summary(self) -> dict[str, Any]:
        total = len(self.pairs)
        by_renderer: dict[str, int] = {}
        accepted = 0
        for pair in self.pairs:
            by_renderer[pair.renderer] = by_renderer.get(pair.renderer, 0) + 1
            if pair.accepted:
                accepted += 1
        return {
            "total": total,
            "accepted": accepted,
            "by_renderer": by_renderer,
            "path": str(self.path),
        }

    def _eligible_pairs(
        self,
        min_quality: float,
        renderer: str | None,
    ) -> list[TrainingPair]:
        return [
            pair
            for pair in self.pairs
            if pair.accepted
            and pair.quality >= min_quality
            and (renderer is None or pair.renderer == renderer)
        ]
