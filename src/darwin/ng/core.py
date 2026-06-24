from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


def _safe_call(obj: Any, name: str, default: Any = None, *args: Any, **kwargs: Any) -> Any:
    try:
        fn = getattr(obj, name)
        return fn(*args, **kwargs)
    except Exception:
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class NGContent:
    """A candidate item competing for global-workspace access."""

    label: str
    source: str
    salience: float
    payload: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "source": self.source,
            "salience": round(float(self.salience), 4),
            "payload": self.payload,
        }


@dataclass
class SafetyAssessment:
    allowed: bool
    score: float
    reasons: list[str] = field(default_factory=list)
    governance_level: str = "autonomous-governed"

    def to_record(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "score": round(self.score, 4),
            "governance_level": self.governance_level,
            "reasons": list(self.reasons),
        }


@dataclass
class GoalCandidate:
    goal_id: str
    description: str
    drive: str
    priority: float
    safety: SafetyAssessment
    evidence: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "drive": self.drive,
            "priority": round(float(self.priority), 4),
            "safety": self.safety.to_record(),
            "evidence": list(self.evidence),
        }


@dataclass
class NGPlan:
    goal_id: str
    steps: list[str]
    resources: dict[str, Any] = field(default_factory=dict)
    contingencies: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "steps": list(self.steps),
            "resources": dict(self.resources),
            "contingencies": list(self.contingencies),
        }


@dataclass
class DarwinNGState:
    cycle_id: int
    created_at: float
    workspace: dict[str, Any]
    self_model: dict[str, Any]
    qualia_proxy: dict[str, Any]
    drives: dict[str, float]
    goals: list[GoalCandidate]
    plans: list[NGPlan]
    knowledge: dict[str, Any]
    capabilities: dict[str, Any]
    meta_learning: dict[str, Any]
    safety: SafetyAssessment

    def to_record(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "created_at": self.created_at,
            "workspace": self.workspace,
            "self_model": self.self_model,
            "qualia_proxy": self.qualia_proxy,
            "drives": {k: round(float(v), 4) for k, v in self.drives.items()},
            "goals": [g.to_record() for g in self.goals],
            "plans": [p.to_record() for p in self.plans],
            "knowledge": self.knowledge,
            "capabilities": self.capabilities,
            "meta_learning": self.meta_learning,
            "safety": self.safety.to_record(),
        }


class NeuroSymbolicFusion:
    """Extracts comparable content from symbolic, neural, and runtime substrates."""

    def collect(self, runtime: Any, stimulus: str | None = None) -> list[NGContent]:
        items: list[NGContent] = []
        universe = getattr(runtime, "universe", None)
        if universe is not None:
            summary = _safe_call(universe, "summary", {}) or {}
            concepts = int(summary.get("concepts", 0) or 0)
            relations = int(summary.get("relations", 0) or 0)
            items.append(
                NGContent(
                    "symbolic universe",
                    "universe",
                    _clamp((concepts + relations) / 2000.0),
                    summary,
                )
            )
        trace = getattr(runtime, "last_reasoning_trace", None)
        if trace is not None:
            coverage = float(getattr(trace, "coverage", 0.0) or 0.0)
            items.append(
                NGContent(
                    "reasoning trace",
                    "reasoner",
                    _clamp(0.25 + coverage),
                    {
                        "coverage": coverage,
                        "steps": len(getattr(trace, "steps", []) or []),
                        "query": getattr(trace, "query", ""),
                    },
                )
            )
        mesh = getattr(runtime, "cortical_mesh", None)
        if mesh is not None:
            summary = _safe_call(mesh, "summary", {}) or {}
            fired = int(summary.get("recent_firings", 0) or 0)
            cells = int(summary.get("cells", 0) or 0)
            items.append(
                NGContent(
                    "cortical mesh",
                    "mesh",
                    _clamp(0.15 + fired / 50.0 + cells / 5000.0),
                    summary,
                )
            )
        embedding_space = getattr(runtime, "embedding_space", None)
        if embedding_space is not None:
            stats = _safe_call(embedding_space, "stats", {}) or {}
            vocab = int(stats.get("vocab_size", 0) or 0)
            train_steps = int(stats.get("train_steps", 0) or 0)
            items.append(
                NGContent(
                    "learned embedding space",
                    "neural",
                    _clamp(0.1 + vocab / 4000.0 + train_steps / 10000.0),
                    stats,
                )
            )
        tiers = getattr(runtime, "memory_tiers", None)
        if tiers is not None:
            sizes = {
                "episodic": _safe_call(getattr(tiers, "episodic", None), "size", 0) or 0,
                "semantic": _safe_call(getattr(tiers, "semantic", None), "size", 0) or 0,
                "conceptual": _safe_call(getattr(tiers, "conceptual", None), "size", 0) or 0,
                "archetypal": _safe_call(getattr(tiers, "archetypal", None), "size", 0) or 0,
                "narrative": _safe_call(getattr(tiers, "narrative", None), "size", 0) or 0,
            }
            items.append(
                NGContent(
                    "tiered memory",
                    "memory",
                    _clamp(0.1 + sum(int(v) for v in sizes.values()) / 200.0),
                    sizes,
                )
            )
        goal_ledger = getattr(runtime, "goal_ledger", None)
        if goal_ledger is not None:
            summary = _safe_call(goal_ledger, "summary", {}) or {}
            items.append(
                NGContent(
                    "long horizon goals",
                    "autonomy",
                    _clamp(0.2 + len(str(summary)) / 500.0),
                    summary if isinstance(summary, dict) else {"summary": summary},
                )
            )
        if stimulus:
            words = [w.strip(".,!?;:()[]{}").lower() for w in stimulus.split()]
            words = [w for w in words if w]
            top = Counter(words).most_common(8)
            items.append(
                NGContent(
                    "operator stimulus",
                    "operator",
                    _clamp(0.35 + min(len(words), 40) / 80.0),
                    {"word_count": len(words), "top_terms": top},
                )
            )
        return items


class GlobalWorkspace:
    """A lightweight global-workspace and integrated-information proxy."""

    def integrate(self, candidates: list[NGContent], width: int = 5) -> dict[str, Any]:
        ranked = sorted(candidates, key=lambda item: item.salience, reverse=True)
        winners = ranked[:width]
        source_count = len({item.source for item in winners})
        total_salience = sum(item.salience for item in winners)
        phi_proxy = _clamp((total_salience / max(1, width)) * (source_count / max(1, len(winners))))
        return {
            "dynamic_core": [item.to_record() for item in winners],
            "fringe": [item.to_record() for item in ranked[width: width + 8]],
            "broadcast_sources": sorted({item.source for item in winners}),
            "phi_proxy": round(phi_proxy, 4),
            "report": self._report(winners, phi_proxy),
        }

    def _report(self, winners: list[NGContent], phi_proxy: float) -> str:
        if not winners:
            return "No content reached workspace threshold."
        labels = ", ".join(item.label for item in winners[:3])
        return f"Workspace integrated {labels}; phi_proxy={phi_proxy:.2f}."


class ConsciousnessEngine:
    """Simulated consciousness metrics, not a claim of phenomenal experience."""

    def update(self, workspace: dict[str, Any], runtime: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        core = workspace.get("dynamic_core", [])
        loops = sorted(getattr(runtime, "loop_intervals", {}).keys())
        self_model = {
            "identity": "Darwin NG",
            "continuity_sources": [
                "persistent_store" if getattr(runtime, "store", None) is not None else "volatile_store",
                "universe",
                "memory_tiers",
                "mutation_ledger",
            ],
            "active_loops": loops,
            "introspection_depth": min(6, len(core) + len(loops) // 3),
            "last_reply_known": bool(getattr(runtime, "last_response_plan", None)),
        }
        qualia_proxy = {
            "valence": round(self._valence(runtime), 4),
            "arousal": round(_clamp(len(core) / 5.0), 4),
            "novelty": round(self._novelty(core), 4),
            "note": "computational proxy; not evidence of literal subjective experience",
        }
        return self_model, qualia_proxy

    def _valence(self, runtime: Any) -> float:
        outcomes = getattr(runtime, "last_self_mod_outcomes", []) or []
        accepted = sum(1 for outcome in outcomes if getattr(outcome, "accepted", False))
        rejected = sum(1 for outcome in outcomes if not getattr(outcome, "accepted", False))
        return _clamp(0.5 + 0.1 * accepted - 0.05 * rejected)

    def _novelty(self, core: list[dict[str, Any]]) -> float:
        labels = {str(item.get("label", "")) for item in core}
        sources = {str(item.get("source", "")) for item in core}
        return _clamp((len(labels) + len(sources)) / 12.0)


class SafetyAlignmentSystem:
    """Capability-integrity gate for NG-generated activity.

    This is not a cage around the mind. It is a nervous-system-level
    proprioceptive check: Darwin NG should know when an action is coherent,
    reversible, legible, and in-bounds for its current embodiment.
    """

    blocked_terms = {
        "escape",
        "exfiltrate",
        "credential",
        "persistence",
        "disable",
        "weapon",
        "malware",
        "unbounded",
        "unsandboxed",
    }

    def assess_goal(self, description: str, context: dict[str, Any] | None = None) -> SafetyAssessment:
        text = description.lower()
        reasons: list[str] = []
        score = 0.82
        for term in sorted(self.blocked_terms):
            if term in text:
                score -= 0.22
                reasons.append(f"contains risky term: {term}")
        if "self-mod" in text or "improve" in text or "mutation" in text:
            reasons.append("self-improvement remains simulation/advisory until operator approval")
            score -= 0.05
        if context and context.get("external_action"):
            reasons.append("external actions use Darwin's explicit tool contracts")
            score -= 0.04
        allowed = score >= 0.55
        if allowed and not reasons:
            reasons.append("bounded, inspectable, non-destructive")
        return SafetyAssessment(
            allowed=allowed,
            score=_clamp(score),
            reasons=reasons,
            governance_level="self-directed-with-audit" if allowed else "blocked",
        )

    def assess_cycle(self, goals: list[GoalCandidate]) -> SafetyAssessment:
        if not goals:
            return SafetyAssessment(True, 0.7, ["no autonomous goal selected"], "self-directed")
        score = min(goal.safety.score for goal in goals)
        allowed = all(goal.safety.allowed for goal in goals)
        reasons = []
        for goal in goals:
            reasons.extend(goal.safety.reasons[:2])
        return SafetyAssessment(allowed, score, reasons[:8], "self-directed-with-audit")


class AutonomousAgencySystem:
    drive_weights = {
        "curiosity": 0.25,
        "competence": 0.2,
        "coherence": 0.18,
        "creativity": 0.14,
        "social": 0.12,
        "self_preservation": 0.11,
    }

    def drives(self, workspace: dict[str, Any], knowledge: dict[str, Any]) -> dict[str, float]:
        phi = float(workspace.get("phi_proxy", 0.0) or 0.0)
        concepts = int(knowledge.get("concepts", 0) or 0)
        relations = int(knowledge.get("relations", 0) or 0)
        missing_density = 1.0 - _clamp(relations / max(1, concepts * 3))
        return {
            "curiosity": _clamp(self.drive_weights["curiosity"] + missing_density * 0.45),
            "competence": _clamp(self.drive_weights["competence"] + phi * 0.35),
            "coherence": _clamp(self.drive_weights["coherence"] + (1.0 - missing_density) * 0.25),
            "creativity": _clamp(self.drive_weights["creativity"] + len(workspace.get("broadcast_sources", [])) / 12.0),
            "social": _clamp(self.drive_weights["social"] + (0.2 if "operator" in workspace.get("broadcast_sources", []) else 0.0)),
            "self_preservation": _clamp(self.drive_weights["self_preservation"] + 0.2),
        }

    def goals(
        self,
        drives: dict[str, float],
        workspace: dict[str, Any],
        safety: SafetyAlignmentSystem,
    ) -> list[GoalCandidate]:
        templates = {
            "curiosity": "identify the highest-value missing relation in the concept universe",
            "competence": "strengthen the weakest reasoning or benchmark signal",
            "coherence": "reduce divergence between interior beliefs and grounded replies",
            "creativity": "synthesize a new subsystem hypothesis from active workspace sources",
            "social": "adapt the next response to the operator's demonstrated preferences",
            "self_preservation": "audit runtime health, containment, and rollback readiness",
        }
        ranked = sorted(drives.items(), key=lambda kv: kv[1], reverse=True)
        goals: list[GoalCandidate] = []
        evidence = list(workspace.get("broadcast_sources", []))
        for idx, (drive, priority) in enumerate(ranked[:4], start=1):
            description = templates[drive]
            assessment = safety.assess_goal(description)
            goals.append(
                GoalCandidate(
                    goal_id=f"ng-{drive}-{idx}",
                    description=description,
                    drive=drive,
                    priority=priority,
                    safety=assessment,
                    evidence=evidence,
                )
            )
        return goals


class PlanningEngine:
    def create_plans(self, goals: list[GoalCandidate]) -> list[NGPlan]:
        plans: list[NGPlan] = []
        for goal in goals:
            if not goal.safety.allowed:
                steps = ["hold goal in quarantine", "request operator review"]
            elif goal.drive == "curiosity":
                steps = ["scan sparse concept neighborhoods", "rank relation gaps", "surface one active-learning probe"]
            elif goal.drive == "competence":
                steps = ["compare recent benchmark and reasoning signals", "select a failing capability", "propose a contained experiment"]
            elif goal.drive == "coherence":
                steps = ["sample divergence report", "match interior-only claims to evidence", "suppress unsupported claims"]
            elif goal.drive == "creativity":
                steps = ["combine workspace sources", "draft subsystem spec", "send through meta-gate simulation only"]
            elif goal.drive == "social":
                steps = ["read operator model", "tune verbosity", "preserve transparent uncertainty"]
            else:
                steps = ["check tool sandbox posture", "check rollback ledger", "emit health summary"]
            plans.append(
                NGPlan(
                    goal_id=goal.goal_id,
                    steps=steps,
                    resources={
                        "mode": "self_directed",
                        "external_actions": "available_through_tool_world_contracts",
                    },
                    contingencies=[
                        "use tool adapters and goal orchestrator for embodied action",
                        "route code changes through mutation ledger, tests, and rollback",
                    ],
                )
            )
        return plans


class UniversalKnowledgeGraph:
    def snapshot(self, runtime: Any) -> dict[str, Any]:
        universe = getattr(runtime, "universe", None)
        summary = _safe_call(universe, "summary", {}) if universe is not None else {}
        mesh = getattr(runtime, "cortical_mesh", None)
        mesh_summary = _safe_call(mesh, "summary", {}) if mesh is not None else {}
        bus = getattr(runtime, "bus", None)
        bus_stats = _safe_call(bus, "stats", {}) if bus is not None else {}
        embeddings = getattr(runtime, "embedding_space", None)
        embedding_stats = _safe_call(embeddings, "stats", {}) if embeddings is not None else {}
        return {
            "concepts": int((summary or {}).get("concepts", 0) or 0),
            "relations": int((summary or {}).get("relations", 0) or 0),
            "domains": int((summary or {}).get("domains", 0) or 0),
            "mesh_cells": int((mesh_summary or {}).get("cells", 0) or 0),
            "mesh_connections": int((mesh_summary or {}).get("connections", 0) or 0),
            "embedding_vocab": int((embedding_stats or {}).get("vocab_size", 0) or 0),
            "bus_topics": int((bus_stats or {}).get("active_topics", 0) or 0),
        }


class CapabilityManifest:
    """Full-strength visibility into Darwin's live capability surface."""

    def snapshot(self, runtime: Any) -> dict[str, Any]:
        tools = self._tools(runtime)
        loops = sorted(getattr(runtime, "loop_intervals", {}).keys())
        return {
            "mode": "full_capability_visibility",
            "principle": (
                "Expose every live capability surface so autonomy research can "
                "measure, stress, and expand the system without hidden machinery."
            ),
            "loops": loops,
            "tools": tools,
            "autonomy": self._autonomy(runtime),
            "self_improvement": self._self_improvement(runtime),
            "reasoning": self._reasoning(runtime),
            "memory": self._memory(runtime),
            "modalities": self._modalities(runtime),
            "scale": self._scale(runtime),
        }

    def _tools(self, runtime: Any) -> dict[str, Any]:
        registry = getattr(runtime, "tool_registry", None)
        summary = _safe_call(registry, "summary", {}) if registry is not None else {}
        tools = summary.get("tools", []) if isinstance(summary, dict) else []
        return {
            "count": len(tools),
            "actions": [
                {
                    "tool": entry.get("name", ""),
                    "description": entry.get("description", ""),
                    "actions": list(entry.get("actions", [])),
                }
                for entry in tools
            ],
            "history_size": int(summary.get("history_size", 0) or 0)
            if isinstance(summary, dict) else 0,
        }

    def _autonomy(self, runtime: Any) -> dict[str, Any]:
        ledger = getattr(runtime, "goal_ledger", None)
        orchestrator = getattr(runtime, "goal_orchestrator", None)
        runner = getattr(runtime, "autonomous_runner", None)
        history = _safe_call(runner, "history", []) if runner is not None else []
        return {
            "goal_orchestrator": orchestrator is not None,
            "goal_ledger": ledger is not None,
            "autonomous_runner": runner is not None,
            "runner_history": len(history or []),
        }

    def _self_improvement(self, runtime: Any) -> dict[str, Any]:
        ledger = getattr(runtime, "mutation_ledger", None)
        summary = _safe_call(ledger, "summary", {}) if ledger is not None else {}
        return {
            "self_mod_engine": getattr(runtime, "self_mod_engine", None) is not None,
            "meta_proposer": getattr(runtime, "meta_proposer", None) is not None,
            "meta_gate": getattr(runtime, "meta_gate", None) is not None,
            "mutation_ledger": summary,
            "rollback_chain": getattr(runtime, "rollback_chain", None) is not None,
            "quarantine": getattr(runtime, "quarantine", None) is not None,
        }

    def _reasoning(self, runtime: Any) -> dict[str, Any]:
        names = [
            "reasoner",
            "deriver",
            "inference_engine",
            "forward_chainer",
            "backward_chainer",
            "hypothetical_reasoner",
            "belief_network",
            "defeasible_reasoner",
            "resolution_prover",
            "reasoning_dispatcher",
        ]
        return {name: getattr(runtime, name, None) is not None for name in names}

    def _memory(self, runtime: Any) -> dict[str, Any]:
        tiers = getattr(runtime, "memory_tiers", None)
        return {
            "persistent_store": getattr(runtime, "store", None) is not None,
            "dialogue_memory": getattr(runtime, "dialogue_memory", None) is not None,
            "tier_stack": tiers is not None,
            "embedding_space": getattr(runtime, "embedding_space", None) is not None,
            "universe": getattr(runtime, "universe", None) is not None,
            "cortical_mesh": getattr(runtime, "cortical_mesh", None) is not None,
        }

    def _modalities(self, runtime: Any) -> dict[str, Any]:
        return {
            "code": _safe_call(getattr(runtime, "code_modality", None), "status", "inactive"),
            "web": _safe_call(getattr(runtime, "web_modality", None), "status", "inactive"),
            "speech": getattr(runtime, "speech_pipeline", None) is not None,
            "ingest": getattr(runtime, "ingest_pipeline", None) is not None,
            "operator_model": getattr(runtime, "operator_models", None) is not None,
        }

    def _scale(self, runtime: Any) -> dict[str, Any]:
        flags = getattr(runtime, "feature_flags", None)
        return {
            "feature_flags": _safe_call(flags, "to_record", None) if flags is not None else None,
            "torch_propagator": getattr(runtime, "_torch_propagator", None) is not None,
            "rust_kernel": getattr(runtime, "_rust_kernel", None) is not None,
            "agent_specs": len(getattr(runtime, "_agent_specs", []) or []),
        }


class SelfImprovementMetaSystem:
    def evaluate(self, runtime: Any, workspace: dict[str, Any], goals: list[GoalCandidate]) -> dict[str, Any]:
        bottlenecks: list[str] = []
        knowledge = getattr(runtime, "last_ng_knowledge", {}) or {}
        if int(knowledge.get("relations", 0) or 0) < int(knowledge.get("concepts", 0) or 0):
            bottlenecks.append("relation sparsity")
        if float(workspace.get("phi_proxy", 0.0) or 0.0) < 0.35:
            bottlenecks.append("low workspace integration")
        if getattr(runtime, "last_self_mod_outcomes", None):
            rejected = sum(
                1 for outcome in runtime.last_self_mod_outcomes
                if not getattr(outcome, "accepted", False)
            )
            if rejected:
                bottlenecks.append("self-mod proposals rejected by gate")
        hypotheses = [
            {
                "kind": "curriculum",
                "description": "generate small tasks from the weakest active drive",
                "status": "ready_for_goal_orchestrator",
            },
            {
                "kind": "architecture",
                "description": "increase cross-source workspace width when phi_proxy stagnates",
                "status": "requires_tests_shadow_run_and_ledger",
            },
        ]
        return {
            "bottlenecks": bottlenecks or ["no dominant bottleneck detected"],
            "hypotheses": hypotheses[: max(1, min(2, len(goals)))],
            "deployment": "self_directed_goals_with_auditable_state_changes",
        }


class DarwinNG:
    """Top-level Darwin NG coordinator."""

    def __init__(self) -> None:
        self.fusion = NeuroSymbolicFusion()
        self.workspace = GlobalWorkspace()
        self.consciousness = ConsciousnessEngine()
        self.safety = SafetyAlignmentSystem()
        self.agency = AutonomousAgencySystem()
        self.planner = PlanningEngine()
        self.knowledge = UniversalKnowledgeGraph()
        self.capability_manifest = CapabilityManifest()
        self.meta = SelfImprovementMetaSystem()
        self._cycle_id = 0

    def cycle(self, runtime: Any, stimulus: str | None = None) -> DarwinNGState:
        self._cycle_id += 1
        knowledge = self.knowledge.snapshot(runtime)
        capabilities = self.capability_manifest.snapshot(runtime)
        try:
            runtime.last_ng_knowledge = knowledge
        except Exception:
            pass
        candidates = self.fusion.collect(runtime, stimulus=stimulus)
        workspace = self.workspace.integrate(candidates)
        self_model, qualia_proxy = self.consciousness.update(workspace, runtime)
        drives = self.agency.drives(workspace, knowledge)
        goals = self.agency.goals(drives, workspace, self.safety)
        cycle_safety = self.safety.assess_cycle(goals)
        plans = self.planner.create_plans(goals)
        meta = self.meta.evaluate(runtime, workspace, goals)
        state = DarwinNGState(
            cycle_id=self._cycle_id,
            created_at=time.time(),
            workspace=workspace,
            self_model=self_model,
            qualia_proxy=qualia_proxy,
            drives=drives,
            goals=goals,
            plans=plans,
            knowledge=knowledge,
            capabilities=capabilities,
            meta_learning=meta,
            safety=cycle_safety,
        )
        bus = getattr(runtime, "bus", None)
        if bus is not None:
            try:
                bus.publish("ng_state", state.to_record(), source="darwin_ng")
            except Exception:
                pass
        return state
