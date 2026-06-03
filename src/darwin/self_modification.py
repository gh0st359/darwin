from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from darwin.types import Transition

if TYPE_CHECKING:
    from darwin.mysterio.meta_gate import MetaGate
    from darwin.mysterio.meta_proposer import MetaProposer, MetaProposerContext
    from darwin.mysterio.proposal_spec import ProposalSpec
    from darwin.mysterio.quarantine import QuarantineQueue
    from darwin.mysterio.snapshot import SnapshotStore


@dataclass
class ProposedModification:
    """A small, testable proposal to tweak Darwin's own machinery."""

    kind: str
    target: str
    rationale: str
    apply: Callable[[Any], None]
    revert: Callable[[Any], None]
    proposal_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    payload: dict[str, Any] = field(default_factory=dict)
    proposed_at: float = field(default_factory=time.time)
    spec: "ProposalSpec | None" = None

    def to_record(self, status: str = "proposed", outcome: dict[str, Any] | None = None) -> dict[str, Any]:
        record = {
            "proposal_id": self.proposal_id,
            "kind": self.kind,
            "target": self.target,
            "rationale": self.rationale,
            "payload": self.payload,
            "status": status,
            "outcome": dict(outcome or {}),
            "proposed_at": self.proposed_at,
        }
        if self.spec is not None:
            record["spec"] = self.spec.to_record()
        return record


@dataclass
class ModificationOutcome:
    proposal: ProposedModification
    accepted: bool
    baseline_error: float
    candidate_error: float
    improvement: float
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        return self.proposal.to_record(
            status="accepted" if self.accepted else "rejected",
            outcome={
                "baseline_error": self.baseline_error,
                "candidate_error": self.candidate_error,
                "improvement": self.improvement,
                "notes": self.notes,
            },
        )


def _prediction_error(darwin: Any, sample: list[Transition]) -> float:
    if not sample:
        return 0.0
    total = 0.0
    counted = 0
    for transition in sample:
        prediction = darwin.causal_model.predict(transition.before, transition.action)
        for variable, observed in dict(transition.after).items():
            if variable in prediction.state:
                expected = prediction.state[variable]
                if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
                    total += abs(float(expected) - float(observed))
                elif expected != observed:
                    total += 1.0
                counted += 1
    if counted == 0:
        return 0.0
    return total / counted


class SelfModificationEngine:
    """Generates, tests, and accepts tweaks to Darwin's own machinery.

    Mysterio extends this with:
      - a typed `ProposalSpec` envelope on every proposal (legacy proposals
        carry `spec=None` and run through a default `PARAMETER` tier);
      - a self-modifiable `MetaGate` that decides acceptance;
      - a plug-in `MetaProposer` that generates structural proposals
        beyond the four hand-authored kinds;
      - a `SnapshotStore` that captures pre-apply state for rollback;
      - a `QuarantineQueue` that tags substrate-touching mutations for
        operator inspection without blocking activation;
      - `TouchRecorder` containment that forces apply() to declare what
        it writes (raises `ContainmentError` on undeclared writes).
    """

    def __init__(
        self,
        darwin: Any,
        holdout_size: int = 12,
        per_cycle_cap: int = 16,
        meta_proposer: "MetaProposer | None" = None,
        meta_gate: "MetaGate | None" = None,
        snapshot_store: "SnapshotStore | None" = None,
        quarantine: "QuarantineQueue | None" = None,
        runtime: Any = None,
        snapshot_capture: Callable[[], Any] | None = None,
    ) -> None:
        self.darwin = darwin
        self.holdout_size = holdout_size
        self.per_cycle_cap = per_cycle_cap
        self.history: list[ModificationOutcome] = []
        self.meta_proposer = meta_proposer
        self.meta_gate = meta_gate
        self.snapshot_store = snapshot_store
        self.quarantine = quarantine
        self.runtime = runtime
        self.snapshot_capture = snapshot_capture

    def propose(self) -> list[ProposedModification]:
        proposals: list[ProposedModification] = []
        proposals.extend(self._propose_min_samples())
        proposals.extend(self._propose_exploration_rate())
        proposals.extend(self._propose_concept_pruning())
        proposals.extend(self._propose_planner_weights())
        if self.meta_proposer is not None:
            from darwin.mysterio.meta_proposer import MetaProposerContext

            ctx = MetaProposerContext(
                darwin=self.darwin,
                runtime=self.runtime,
                recent_outcomes=list(self.history[-32:]),
                last_simulation=getattr(self.runtime, "last_simulation", None),
                last_uncertainty_scan=getattr(self.runtime, "last_uncertainty_scan", None),
            )
            proposals.extend(self.meta_proposer.propose(ctx))
        return proposals

    def evaluate(self, proposal: ProposedModification) -> ModificationOutcome:
        sample = self._holdout_sample()
        baseline_error = _prediction_error(self.darwin, sample)

        snapshot = self._snapshot()
        pre_mind_snapshot_id = self._capture_mind_snapshot()
        try:
            self._apply_with_containment(proposal)
            candidate_error = _prediction_error(self.darwin, sample)
            improvement = baseline_error - candidate_error
            accepted = self._gate_decision(
                proposal, improvement, baseline_error, candidate_error
            )
            notes = "accepted in self-test" if accepted else "rejected: gate"
            if not accepted:
                proposal.revert(self.darwin)
                self._restore(snapshot)
            outcome = ModificationOutcome(
                proposal=proposal,
                accepted=accepted,
                baseline_error=baseline_error,
                candidate_error=candidate_error,
                improvement=improvement,
                notes=notes,
            )
        except Exception as exc:
            try:
                proposal.revert(self.darwin)
            except Exception:
                pass
            self._restore(snapshot)
            outcome = ModificationOutcome(
                proposal=proposal,
                accepted=False,
                baseline_error=baseline_error,
                candidate_error=baseline_error,
                improvement=0.0,
                notes=f"reverted after exception: {exc!r}",
            )
        self.history.append(outcome)
        if outcome.accepted and self.quarantine is not None and proposal.spec is not None:
            try:
                self.quarantine.submit(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.spec.kind,
                    description=proposal.spec.description or proposal.rationale,
                    snapshot_id=pre_mind_snapshot_id or "",
                    notes=outcome.notes,
                    extra={"signature": proposal.spec.introspection_signature},
                )
            except Exception:
                pass
        return outcome

    def run_cycle(self) -> list[ModificationOutcome]:
        proposals = self.propose()
        outcomes: list[ModificationOutcome] = []
        for proposal in proposals[: self.per_cycle_cap]:
            outcomes.append(self.evaluate(proposal))
        return outcomes

    # -- internal helpers ----------------------------------------------------

    def _apply_with_containment(self, proposal: ProposedModification) -> None:
        spec = proposal.spec
        if spec is None or not spec.touches:
            proposal.apply(self.darwin)
            return
        # Live containment: register every declared touch target with the
        # recorder, run apply inside its context manager. Undeclared writes
        # to any registered target raise ContainmentError, which propagates
        # back through evaluate() so the outcome is recorded as failed and
        # the snapshot is rolled back.
        from darwin.mysterio.safety import TouchRecorder

        recorder = TouchRecorder(spec.touches)
        for path, target in self._resolve_touch_targets(spec.touches).items():
            recorder.register(path, target)
        with recorder:
            proposal.apply(self.darwin)

    def _resolve_touch_targets(self, touches: set[str]) -> dict[str, Any]:
        """Resolve declared touch paths (e.g. "darwin.memory.episodes") into
        live object references for the recorder to intercept.

        Each touch path is "<root>.<attribute>"; the root is resolved against
        ``self.darwin`` (or ``self.runtime`` if available) and the attribute
        is the write-target. Returns a dict keyed by the root path.
        """

        resolved: dict[str, Any] = {}
        for path in touches:
            parts = path.split(".")
            if len(parts) < 2:
                continue
            root_name = parts[0]
            if root_name in resolved:
                continue
            target: Any = None
            if root_name == "darwin":
                target = self.darwin
            elif root_name == "runtime" and self.runtime is not None:
                target = self.runtime
            elif root_name == "universe" and getattr(self.darwin, "universe", None):
                target = self.darwin.universe
            elif root_name == "memory" and getattr(self.darwin, "memory", None):
                target = self.darwin.memory
            else:
                target = getattr(self.darwin, root_name, None)
                if target is None and self.runtime is not None:
                    target = getattr(self.runtime, root_name, None)
            if target is None:
                continue
            # Walk intermediate attribute hops so the recorder sees the
            # final container (so writes to ``darwin.memory.episodes`` are
            # caught even when only ``darwin.memory`` is the registered
            # interception target).
            current: Any = target
            full_path = root_name
            for attr in parts[1:-1]:
                current = getattr(current, attr, None)
                if current is None:
                    break
                full_path = f"{full_path}.{attr}"
                if full_path not in resolved:
                    resolved[full_path] = current
            resolved.setdefault(root_name, target)
        return resolved

    def _gate_decision(
        self,
        proposal: ProposedModification,
        improvement: float,
        baseline_error: float,
        candidate_error: float,
    ) -> bool:
        if self.meta_gate is None:
            return improvement > 0.0 and candidate_error <= baseline_error
        from darwin.mysterio.meta_gate import GateInputs

        inputs = GateInputs(
            improvement=improvement,
            baseline_error=baseline_error,
            candidate_error=candidate_error,
            continuity_term=0.0,
            visibility_term=0.0,
        )
        decision = self.meta_gate.decide(inputs)
        return decision.accepted

    def _capture_mind_snapshot(self) -> str | None:
        if self.snapshot_store is None:
            return None
        try:
            if self.snapshot_capture is not None:
                snapshot = self.snapshot_capture()
            else:
                from darwin.mysterio.snapshot import MindSnapshot

                gate_id = self.meta_gate.current.gate_id if self.meta_gate else "default"
                snapshot = MindSnapshot.capture(
                    self.darwin,
                    gate_identity=gate_id,
                    self_mod_history_len=len(self.history),
                )
            return self.snapshot_store.record(snapshot)
        except Exception:
            return None

    def _holdout_sample(self) -> list[Transition]:
        return list(self.darwin.memory.episodes.recent(self.holdout_size))

    def _snapshot(self) -> dict[str, Any]:
        causal = self.darwin.causal_model
        return {
            "min_samples": causal.min_samples,
            "exploration_rate": self.darwin.exploration_rate,
            "planner": getattr(self.darwin, "_planner_overrides", {}).copy(),
        }

    def _restore(self, snapshot: dict[str, Any]) -> None:
        causal = self.darwin.causal_model
        causal.min_samples = snapshot["min_samples"]
        self.darwin.exploration_rate = snapshot["exploration_rate"]
        if hasattr(self.darwin, "_planner_overrides"):
            self.darwin._planner_overrides = dict(snapshot.get("planner", {}))

    def _propose_min_samples(self) -> list[ProposedModification]:
        causal = self.darwin.causal_model
        current = causal.min_samples
        choices = []
        if current > 2:
            new_value = current - 1
            choices.append((new_value, "lower min_samples for faster belief crystallization"))
        if current < 8:
            new_value = current + 1
            choices.append((new_value, "raise min_samples to demand more evidence per belief"))
        return [self._make_min_samples_proposal(value, rationale) for value, rationale in choices]

    def _make_min_samples_proposal(self, new_value: int, rationale: str) -> ProposedModification:
        old_value = self.darwin.causal_model.min_samples

        def apply(darwin: Any) -> None:
            darwin.causal_model.min_samples = new_value

        def revert(darwin: Any) -> None:
            darwin.causal_model.min_samples = old_value

        return ProposedModification(
            kind="causal.min_samples",
            target="causal_model.min_samples",
            rationale=rationale,
            apply=apply,
            revert=revert,
            payload={"old": old_value, "new": new_value},
        )

    def _propose_exploration_rate(self) -> list[ProposedModification]:
        old_rate = self.darwin.exploration_rate
        proposals: list[ProposedModification] = []
        for delta, label in [(-0.05, "less exploration"), (0.05, "more exploration")]:
            new_rate = max(0.0, min(0.6, old_rate + delta))
            if abs(new_rate - old_rate) < 1e-6:
                continue

            def apply(darwin: Any, rate: float = new_rate) -> None:
                darwin.exploration_rate = rate

            def revert(darwin: Any, rate: float = old_rate) -> None:
                darwin.exploration_rate = rate

            proposals.append(
                ProposedModification(
                    kind="exploration.rate",
                    target="darwin.exploration_rate",
                    rationale=f"{label} based on current uncertainty",
                    apply=apply,
                    revert=revert,
                    payload={"old": old_rate, "new": new_rate},
                )
            )
        return proposals

    def _propose_concept_pruning(self) -> list[ProposedModification]:
        concept_index = self.darwin.memory.concepts
        candidates = [
            concept
            for concept in concept_index._concepts.values()
            if concept.support <= 1 and concept.level <= 1
        ]
        if len(candidates) < 5:
            return []
        target_names = [concept.name for concept in candidates[:5]]
        backup = {name: copy.deepcopy(concept_index._concepts[name]) for name in target_names}

        def apply(darwin: Any) -> None:
            for name in target_names:
                darwin.memory.concepts._concepts.pop(name, None)

        def revert(darwin: Any) -> None:
            for name, original in backup.items():
                darwin.memory.concepts._concepts[name] = original

        return [
            ProposedModification(
                kind="concept.prune",
                target="memory.concepts",
                rationale="prune low-support, low-level concepts that may be noise",
                apply=apply,
                revert=revert,
                payload={"pruned": target_names},
            )
        ]

    def _propose_planner_weights(self) -> list[ProposedModification]:
        overrides = getattr(self.darwin, "_planner_overrides", {})
        current_curiosity = float(overrides.get("exploration_bias", 1.0))
        proposals: list[ProposedModification] = []
        for delta, label in [(-0.2, "reduce curiosity bias"), (0.2, "increase curiosity bias")]:
            new_value = max(0.4, min(2.0, current_curiosity + delta))
            if abs(new_value - current_curiosity) < 1e-6:
                continue

            def apply(darwin: Any, value: float = new_value) -> None:
                store = getattr(darwin, "_planner_overrides", None)
                if store is None:
                    store = {}
                    setattr(darwin, "_planner_overrides", store)
                store["exploration_bias"] = value

            def revert(darwin: Any, value: float = current_curiosity) -> None:
                store = getattr(darwin, "_planner_overrides", None)
                if store is None:
                    return
                store["exploration_bias"] = value

            proposals.append(
                ProposedModification(
                    kind="planner.exploration_bias",
                    target="planner.exploration_bias",
                    rationale=label,
                    apply=apply,
                    revert=revert,
                    payload={"old": current_curiosity, "new": new_value},
                )
            )
        return proposals
