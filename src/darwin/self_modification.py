from __future__ import annotations

import copy
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from darwin.types import Transition


# ---------------------------------------------------------------------------
# Proposal + outcome types.
# ---------------------------------------------------------------------------


@dataclass
class ProposedModification:
    """A small, testable proposal to tweak Darwin's own machinery.

    Phase E makes proposals **declarative**: every proposal carries a
    ``kind`` + ``payload`` that the ``_PROPOSAL_REGISTRY`` below can
    reconstruct into apply/revert closures from the payload alone. The
    closure-form fields below are still used in-process; the registry
    mirror is what persists across restarts so accepted mods can replay.
    """

    kind: str
    target: str
    rationale: str
    apply: Callable[[Any], None]
    revert: Callable[[Any], None]
    proposal_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    payload: dict[str, Any] = field(default_factory=dict)
    proposed_at: float = field(default_factory=time.time)

    def to_record(self, status: str = "proposed", outcome: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "kind": self.kind,
            "target": self.target,
            "rationale": self.rationale,
            "payload": self.payload,
            "status": status,
            "outcome": dict(outcome or {}),
            "proposed_at": self.proposed_at,
        }


@dataclass
class ModificationOutcome:
    proposal: ProposedModification
    accepted: bool
    baseline_error: float
    candidate_error: float
    improvement: float
    ci_low: float = 0.0
    ci_high: float = 0.0
    relative_improvement: float = 0.0
    sample_size: int = 0
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        return self.proposal.to_record(
            status="accepted" if self.accepted else "rejected",
            outcome={
                "baseline_error": self.baseline_error,
                "candidate_error": self.candidate_error,
                "improvement": self.improvement,
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
                "relative_improvement": self.relative_improvement,
                "sample_size": self.sample_size,
                "notes": self.notes,
            },
        )


# ---------------------------------------------------------------------------
# Error metric (per-sample for bootstrap support).
# ---------------------------------------------------------------------------


def _prediction_errors(darwin: Any, sample: list[Transition]) -> list[float]:
    """Per-sample mean absolute prediction error.

    Returns one error value per transition; ``mean(...)`` of the result
    recovers the legacy scalar. The Phase E accept gate runs a paired
    bootstrap on per-sample deltas, so we need the vector form.
    """

    errors: list[float] = []
    for transition in sample:
        prediction = darwin.causal_model.predict(transition.before, transition.action)
        total = 0.0
        counted = 0
        for variable, observed in dict(transition.after).items():
            if variable not in prediction.state:
                continue
            expected = prediction.state[variable]
            if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
                total += abs(float(expected) - float(observed))
            elif expected != observed:
                total += 1.0
            counted += 1
        errors.append(total / counted if counted else 0.0)
    return errors


def _prediction_error(darwin: Any, sample: list[Transition]) -> float:
    errors = _prediction_errors(darwin, sample)
    if not errors:
        return 0.0
    return sum(errors) / len(errors)


# ---------------------------------------------------------------------------
# Paired bootstrap accept gate.
# ---------------------------------------------------------------------------


def paired_bootstrap_ci(
    deltas: list[float],
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Return (point estimate, ci_low, ci_high) of the mean of ``deltas``.

    Implements a stdlib-only paired bootstrap. With 1000 resamples and
    confidence=0.95 the CI uses the 2.5th and 97.5th percentiles of the
    resampled means.
    """

    if not deltas:
        return 0.0, 0.0, 0.0
    point = sum(deltas) / len(deltas)
    rng = random.Random(seed if seed is not None else 0xD4)
    n = len(deltas)
    means: list[float] = []
    for _ in range(resamples):
        sampled = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sampled) / n)
    means.sort()
    alpha = 1.0 - confidence
    lower_idx = max(0, int(alpha / 2.0 * resamples) - 1)
    upper_idx = min(resamples - 1, int((1.0 - alpha / 2.0) * resamples) - 1)
    return point, means[lower_idx], means[upper_idx]


# ---------------------------------------------------------------------------
# Declarative proposal registry — reconstructs apply/revert from (kind, payload).
# ---------------------------------------------------------------------------


_PROPOSAL_REGISTRY: dict[str, dict[str, Callable[..., Callable[[Any], None]]]] = {}


def register_proposal_kind(
    kind: str,
    apply_factory: Callable[[dict[str, Any]], Callable[[Any], None]],
    revert_factory: Callable[[dict[str, Any]], Callable[[Any], None]],
    replayable: bool = True,
) -> None:
    """Register a proposal kind so it can be replayed from the ledger."""

    _PROPOSAL_REGISTRY[kind] = {
        "apply": apply_factory,
        "revert": revert_factory,
        "replayable": replayable,  # type: ignore[assignment]
    }


def _apply_min_samples(payload: dict[str, Any]) -> Callable[[Any], None]:
    new_value = int(payload["new"])
    def _apply(darwin: Any) -> None:
        darwin.causal_model.min_samples = new_value
    return _apply


def _revert_min_samples(payload: dict[str, Any]) -> Callable[[Any], None]:
    old_value = int(payload["old"])
    def _revert(darwin: Any) -> None:
        darwin.causal_model.min_samples = old_value
    return _revert


def _apply_exploration_rate(payload: dict[str, Any]) -> Callable[[Any], None]:
    new_value = float(payload["new"])
    def _apply(darwin: Any) -> None:
        darwin.exploration_rate = new_value
    return _apply


def _revert_exploration_rate(payload: dict[str, Any]) -> Callable[[Any], None]:
    old_value = float(payload["old"])
    def _revert(darwin: Any) -> None:
        darwin.exploration_rate = old_value
    return _revert


def _apply_curiosity_bias(payload: dict[str, Any]) -> Callable[[Any], None]:
    new_value = float(payload["new"])
    def _apply(darwin: Any) -> None:
        store = getattr(darwin, "_planner_overrides", None)
        if store is None:
            store = {}
            setattr(darwin, "_planner_overrides", store)
        store["exploration_bias"] = new_value
    return _apply


def _revert_curiosity_bias(payload: dict[str, Any]) -> Callable[[Any], None]:
    old_value = float(payload["old"])
    def _revert(darwin: Any) -> None:
        store = getattr(darwin, "_planner_overrides", None)
        if store is None:
            return
        store["exploration_bias"] = old_value
    return _revert


def _apply_realizer_param(payload: dict[str, Any]) -> Callable[[Any], None]:
    field_name = str(payload["field"])
    new_value = float(payload["new"])
    def _apply(darwin: Any) -> None:
        runtime = getattr(darwin, "_runtime_ref", None)
        if runtime is None:
            return
        dlm = getattr(runtime, "dlm", None)
        config = getattr(dlm, "config", None)
        if config is None:
            return
        setattr(config, field_name, new_value)
    return _apply


def _revert_realizer_param(payload: dict[str, Any]) -> Callable[[Any], None]:
    field_name = str(payload["field"])
    old_value = float(payload["old"])
    def _revert(darwin: Any) -> None:
        runtime = getattr(darwin, "_runtime_ref", None)
        if runtime is None:
            return
        dlm = getattr(runtime, "dlm", None)
        config = getattr(dlm, "config", None)
        if config is None:
            return
        setattr(config, field_name, old_value)
    return _revert


def _apply_kernel_priority_weight(payload: dict[str, Any]) -> Callable[[Any], None]:
    weight_name = str(payload["weight"])
    new_value = float(payload["new"])
    def _apply(darwin: Any) -> None:
        runtime = getattr(darwin, "_runtime_ref", None)
        driver = getattr(runtime, "_kernel_driver", None) if runtime is not None else None
        if driver is None:
            return
        driver.priority_weights[weight_name] = new_value
    return _apply


def _revert_kernel_priority_weight(payload: dict[str, Any]) -> Callable[[Any], None]:
    weight_name = str(payload["weight"])
    old_value = float(payload["old"])
    def _revert(darwin: Any) -> None:
        runtime = getattr(darwin, "_runtime_ref", None)
        driver = getattr(runtime, "_kernel_driver", None) if runtime is not None else None
        if driver is None:
            return
        driver.priority_weights[weight_name] = old_value
    return _revert


# Register the kinds.
register_proposal_kind("causal.min_samples", _apply_min_samples, _revert_min_samples)
register_proposal_kind("exploration.rate", _apply_exploration_rate, _revert_exploration_rate)
register_proposal_kind("planner.exploration_bias", _apply_curiosity_bias, _revert_curiosity_bias)
register_proposal_kind("realizer.config", _apply_realizer_param, _revert_realizer_param)
register_proposal_kind("kernel.priority_weight", _apply_kernel_priority_weight, _revert_kernel_priority_weight)


# ---------------------------------------------------------------------------
# Tunable bounds (used by both safety_rejection and replay quarantine).
# ---------------------------------------------------------------------------


SAFETY_BOUNDS: dict[str, tuple[float, float]] = {
    "causal.min_samples": (3, 12),
    "exploration.rate": (0.05, 0.6),
    "planner.exploration_bias": (0.5, 2.5),
    "realizer.config:connector_frequency": (0.0, 1.0),
    "realizer.config:aside_rate": (0.0, 1.0),
    "realizer.config:qualifier_strength": (0.0, 1.0),
    "kernel.priority_weight:uncertainty": (0.1, 1.0),
    "kernel.priority_weight:learning_priority_match": (0.0, 1.0),
    "kernel.priority_weight:age": (0.0, 1.0),
}


def _within_bounds(kind: str, payload: dict[str, Any]) -> bool:
    if kind in SAFETY_BOUNDS:
        lo, hi = SAFETY_BOUNDS[kind]
        try:
            return lo <= float(payload.get("new")) <= hi
        except (TypeError, ValueError):
            return False
    bound_key = None
    if kind == "realizer.config":
        bound_key = f"realizer.config:{payload.get('field')}"
    elif kind == "kernel.priority_weight":
        bound_key = f"kernel.priority_weight:{payload.get('weight')}"
    if bound_key and bound_key in SAFETY_BOUNDS:
        lo, hi = SAFETY_BOUNDS[bound_key]
        try:
            return lo <= float(payload.get("new")) <= hi
        except (TypeError, ValueError):
            return False
    return True


# ---------------------------------------------------------------------------
# SelfModificationEngine.
# ---------------------------------------------------------------------------


class SelfModificationEngine:
    """Generates, tests, accepts, persists, and replays Darwin's self-tweaks.

    Phase E rewrites the accept gate. v4 demanded ``improvement > 0.0``
    strict on a 12-sample holdout; with integer-delta errors the
    difference almost never exceeded zero, so every proposal rejected.

    The v5 gate:
      - Bumps holdout to 64 (caps at available episodes).
      - Computes per-sample baseline + candidate errors.
      - Runs a 1000-resample paired bootstrap on deltas.
      - Accepts if 95% CI of mean delta is strictly above zero OR if
        the relative improvement exceeds 5%.
      - Persists every accepted mod to the self_mod_ledger table.
      - On boot, ``replay_ledger()`` re-applies every accepted mod.
    """

    MIN_CAUSAL_SAMPLES = 3
    MIN_EXPLORATION_RATE = 0.05
    MIN_CURIOSITY_BIAS = 0.5
    BOOTSTRAP_RESAMPLES = 1000
    BOOTSTRAP_CONFIDENCE = 0.95
    RELATIVE_IMPROVEMENT_THRESHOLD = 0.05

    def __init__(
        self,
        darwin: Any,
        holdout_size: int = 64,
        runtime: Any | None = None,
        store: Any | None = None,
    ) -> None:
        self.darwin = darwin
        self.holdout_size = holdout_size
        self.history: list[ModificationOutcome] = []
        self.runtime = runtime
        self.store = store or getattr(darwin, "store", None)
        # Make runtime reachable from declarative apply/revert closures
        # (they expect ``darwin._runtime_ref``).
        if runtime is not None:
            try:
                setattr(darwin, "_runtime_ref", runtime)
            except Exception:
                pass

    def attach_runtime(self, runtime: Any) -> None:
        self.runtime = runtime
        try:
            setattr(self.darwin, "_runtime_ref", runtime)
        except Exception:
            pass

    # -- proposals ---------------------------------------------------------

    def propose(self) -> list[ProposedModification]:
        proposals: list[ProposedModification] = []
        proposals.extend(self._propose_min_samples())
        proposals.extend(self._propose_exploration_rate())
        proposals.extend(self._propose_concept_pruning())
        proposals.extend(self._propose_planner_weights())
        proposals.extend(self._propose_realizer_config())
        proposals.extend(self._propose_kernel_priority_weights())
        return proposals

    def evaluate(self, proposal: ProposedModification) -> ModificationOutcome:
        sample = self._holdout_sample()
        baseline_errors = _prediction_errors(self.darwin, sample)
        baseline_mean = sum(baseline_errors) / len(baseline_errors) if baseline_errors else 0.0

        snapshot = self._snapshot()
        try:
            safety_note = self._safety_rejection(proposal)
            if safety_note:
                return self._reject_without_applying(proposal, baseline_mean, safety_note)
            proposal.apply(self.darwin)
            candidate_errors = _prediction_errors(self.darwin, sample)
            candidate_mean = sum(candidate_errors) / len(candidate_errors) if candidate_errors else 0.0
            deltas = [b - c for b, c in zip(baseline_errors, candidate_errors)]
            improvement, ci_low, ci_high = paired_bootstrap_ci(
                deltas,
                resamples=self.BOOTSTRAP_RESAMPLES,
                confidence=self.BOOTSTRAP_CONFIDENCE,
                seed=hash(proposal.proposal_id) & 0xFFFFFFFF,
            )
            relative = (baseline_mean - candidate_mean) / baseline_mean if baseline_mean > 0 else 0.0
            ci_supports = ci_low > 0.0
            relative_supports = relative >= self.RELATIVE_IMPROVEMENT_THRESHOLD
            accepted = ci_supports or relative_supports
            if not accepted:
                proposal.revert(self.darwin)
                self._restore(snapshot)
            notes = (
                f"accepted via CI ({ci_low:.4f}..{ci_high:.4f})" if ci_supports
                else f"accepted via relative {relative:.3f}" if relative_supports
                else f"rejected: CI {ci_low:.4f}..{ci_high:.4f}, rel {relative:.3f}"
            )
            outcome = ModificationOutcome(
                proposal=proposal,
                accepted=accepted,
                baseline_error=baseline_mean,
                candidate_error=candidate_mean,
                improvement=improvement,
                ci_low=ci_low,
                ci_high=ci_high,
                relative_improvement=relative,
                sample_size=len(sample),
                notes=notes,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._restore(snapshot)
            outcome = ModificationOutcome(
                proposal=proposal,
                accepted=False,
                baseline_error=baseline_mean,
                candidate_error=baseline_mean,
                improvement=0.0,
                sample_size=len(sample),
                notes=f"reverted after exception: {exc!r}",
            )
        self.history.append(outcome)
        # Persist to the v5 ledger if a store is attached.
        if self.store is not None and hasattr(self.store, "record_self_mod"):
            try:
                self.store.record_self_mod(outcome)
            except Exception:
                pass
        return outcome

    def run_cycle(self) -> list[ModificationOutcome]:
        proposals = self.propose()
        outcomes: list[ModificationOutcome] = []
        for proposal in proposals[:3]:
            outcomes.append(self.evaluate(proposal))
        return outcomes

    # -- ledger replay ----------------------------------------------------

    def replay_ledger(self) -> dict[str, int]:
        """Re-apply every accepted self-mod from the ledger on startup.

        Returns a small summary dict: ``{"applied": N, "quarantined": M,
        "skipped": K}``.
        """

        store = self.store
        summary = {"applied": 0, "quarantined": 0, "skipped": 0}
        if store is None or not hasattr(store, "list_self_mods"):
            return summary
        try:
            rows = store.list_self_mods(limit=10_000, status="accepted")
        except Exception:
            return summary
        # Replay in chronological order — older mods first.
        rows.sort(key=lambda row: row.get("applied_at") or 0.0)
        for row in rows:
            kind = row.get("kind", "")
            payload = row.get("payload", {}) or {}
            if not isinstance(payload, dict):
                summary["skipped"] += 1
                continue
            if not _within_bounds(kind, payload):
                summary["quarantined"] += 1
                if hasattr(store, "mark_self_mod_quarantined"):
                    try:
                        store.mark_self_mod_quarantined(row.get("proposal_id", ""))
                    except Exception:
                        pass
                continue
            handler = _PROPOSAL_REGISTRY.get(kind)
            if handler is None:
                summary["skipped"] += 1
                continue
            try:
                handler["apply"](payload)(self.darwin)
                summary["applied"] += 1
            except Exception:
                summary["skipped"] += 1
        return summary

    # -- holdout & snapshots ----------------------------------------------

    def _holdout_sample(self) -> list[Transition]:
        return list(self.darwin.memory.episodes.recent(self.holdout_size))

    def _snapshot(self) -> dict[str, Any]:
        causal = self.darwin.causal_model
        snapshot = {
            "min_samples": causal.min_samples,
            "exploration_rate": self.darwin.exploration_rate,
            "planner": getattr(self.darwin, "_planner_overrides", {}).copy(),
        }
        runtime = getattr(self.darwin, "_runtime_ref", None) or self.runtime
        if runtime is not None:
            dlm = getattr(runtime, "dlm", None)
            config = getattr(dlm, "config", None)
            if config is not None:
                snapshot["realizer_config"] = {
                    "connector_frequency": getattr(config, "connector_frequency", 0.0),
                    "aside_rate": getattr(config, "aside_rate", 0.0),
                    "qualifier_strength": getattr(config, "qualifier_strength", 0.0),
                }
            driver = getattr(runtime, "_kernel_driver", None)
            if driver is not None:
                snapshot["kernel_weights"] = dict(driver.priority_weights)
        return snapshot

    def _restore(self, snapshot: dict[str, Any]) -> None:
        causal = self.darwin.causal_model
        causal.min_samples = snapshot["min_samples"]
        self.darwin.exploration_rate = snapshot["exploration_rate"]
        if hasattr(self.darwin, "_planner_overrides"):
            self.darwin._planner_overrides = dict(snapshot.get("planner", {}))
        runtime = getattr(self.darwin, "_runtime_ref", None) or self.runtime
        if runtime is not None and "realizer_config" in snapshot:
            dlm = getattr(runtime, "dlm", None)
            config = getattr(dlm, "config", None)
            if config is not None:
                for key, value in snapshot["realizer_config"].items():
                    setattr(config, key, value)
        if runtime is not None and "kernel_weights" in snapshot:
            driver = getattr(runtime, "_kernel_driver", None)
            if driver is not None:
                driver.priority_weights = dict(snapshot["kernel_weights"])

    # -- proposal generators ----------------------------------------------

    def _propose_min_samples(self) -> list[ProposedModification]:
        causal = self.darwin.causal_model
        current = causal.min_samples
        choices = []
        if current > self.MIN_CAUSAL_SAMPLES:
            new_value = current - 1
            choices.append((new_value, "lower min_samples for faster belief crystallization"))
        if current < 8:
            new_value = current + 1
            choices.append((new_value, "raise min_samples to demand more evidence per belief"))
        return [self._make_min_samples_proposal(value, rationale) for value, rationale in choices]

    def _make_min_samples_proposal(self, new_value: int, rationale: str) -> ProposedModification:
        old_value = self.darwin.causal_model.min_samples
        payload = {"old": old_value, "new": new_value}
        return ProposedModification(
            kind="causal.min_samples",
            target="causal_model.min_samples",
            rationale=rationale,
            apply=_apply_min_samples(payload),
            revert=_revert_min_samples(payload),
            payload=payload,
        )

    def _propose_exploration_rate(self) -> list[ProposedModification]:
        old_rate = self.darwin.exploration_rate
        proposals: list[ProposedModification] = []
        for delta, label in [(-0.05, "less exploration"), (0.05, "more exploration")]:
            new_rate = max(self.MIN_EXPLORATION_RATE, min(0.6, old_rate + delta))
            if abs(new_rate - old_rate) < 1e-6:
                continue
            payload = {"old": old_rate, "new": new_rate}
            proposals.append(
                ProposedModification(
                    kind="exploration.rate",
                    target="darwin.exploration_rate",
                    rationale=f"{label} based on current uncertainty",
                    apply=_apply_exploration_rate(payload),
                    revert=_revert_exploration_rate(payload),
                    payload=payload,
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

        # concept.prune isn't replayable from payload because original
        # concept objects aren't reconstructible from the ledger row. We
        # don't register it in _PROPOSAL_REGISTRY for that reason.
        return [
            ProposedModification(
                kind="concept.prune",
                target="memory.concepts",
                rationale="prune low-support, low-level concepts that may be noise",
                apply=apply,
                revert=revert,
                payload={"pruned": target_names, "replayable": False},
            )
        ]

    def _propose_planner_weights(self) -> list[ProposedModification]:
        overrides = getattr(self.darwin, "_planner_overrides", {})
        current_curiosity = float(overrides.get("exploration_bias", 1.0))
        proposals: list[ProposedModification] = []
        for delta, label in [(-0.2, "reduce curiosity bias"), (0.2, "increase curiosity bias")]:
            new_value = max(self.MIN_CURIOSITY_BIAS, min(2.5, current_curiosity + delta))
            if abs(new_value - current_curiosity) < 1e-6:
                continue
            payload = {"old": current_curiosity, "new": new_value}
            proposals.append(
                ProposedModification(
                    kind="planner.exploration_bias",
                    target="planner.exploration_bias",
                    rationale=label,
                    apply=_apply_curiosity_bias(payload),
                    revert=_revert_curiosity_bias(payload),
                    payload=payload,
                )
            )
        return proposals

    def _propose_realizer_config(self) -> list[ProposedModification]:
        """Phase E new kind: tune the symbolic realizer's voice parameters.

        Only fires when a runtime + DLM with a ``config`` attribute is
        reachable (i.e. the v5 SymbolicRealizerDLM is in use).
        """

        runtime = self.runtime or getattr(self.darwin, "_runtime_ref", None)
        dlm = getattr(runtime, "dlm", None) if runtime is not None else None
        config = getattr(dlm, "config", None)
        if config is None:
            return []
        proposals: list[ProposedModification] = []
        for field_name, delta in [
            ("connector_frequency", 0.05),
            ("aside_rate", 0.05),
            ("qualifier_strength", 0.05),
        ]:
            current = float(getattr(config, field_name))
            for direction, label in [(-1, "loosen"), (1, "tighten")]:
                new_value = max(0.0, min(1.0, current + direction * delta))
                if abs(new_value - current) < 1e-6:
                    continue
                payload = {"old": current, "new": new_value, "field": field_name}
                proposals.append(
                    ProposedModification(
                        kind="realizer.config",
                        target=f"realizer.config.{field_name}",
                        rationale=f"{label} {field_name} by {delta}",
                        apply=_apply_realizer_param(payload),
                        revert=_revert_realizer_param(payload),
                        payload=payload,
                    )
                )
        return proposals

    def _propose_kernel_priority_weights(self) -> list[ProposedModification]:
        runtime = self.runtime or getattr(self.darwin, "_runtime_ref", None)
        driver = getattr(runtime, "_kernel_driver", None) if runtime is not None else None
        if driver is None:
            return []
        proposals: list[ProposedModification] = []
        for weight_name, delta in [("uncertainty", 0.05), ("learning_priority_match", 0.05)]:
            current = float(driver.priority_weights.get(weight_name, 0.0))
            for direction, label in [(-1, "tilt away"), (1, "tilt toward")]:
                bound_key = f"kernel.priority_weight:{weight_name}"
                lo, hi = SAFETY_BOUNDS.get(bound_key, (0.0, 1.0))
                new_value = max(lo, min(hi, current + direction * delta))
                if abs(new_value - current) < 1e-6:
                    continue
                payload = {"old": current, "new": new_value, "weight": weight_name}
                proposals.append(
                    ProposedModification(
                        kind="kernel.priority_weight",
                        target=f"kernel.priority_weights.{weight_name}",
                        rationale=f"{label} {weight_name} by {delta}",
                        apply=_apply_kernel_priority_weight(payload),
                        revert=_revert_kernel_priority_weight(payload),
                        payload=payload,
                    )
                )
        return proposals

    # -- safety -----------------------------------------------------------

    def _safety_rejection(self, proposal: ProposedModification) -> str:
        if not _within_bounds(proposal.kind, proposal.payload):
            return f"rejected: {proposal.kind} payload outside declared bounds"
        return ""

    def _reject_without_applying(
        self,
        proposal: ProposedModification,
        baseline_error: float,
        notes: str,
    ) -> ModificationOutcome:
        outcome = ModificationOutcome(
            proposal=proposal,
            accepted=False,
            baseline_error=baseline_error,
            candidate_error=baseline_error,
            improvement=0.0,
            sample_size=0,
            notes=notes,
        )
        self.history.append(outcome)
        return outcome
