from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any


def _id(*parts: Any, prefix: str = "strat") -> str:
    digest = hashlib.sha256("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class StrategicObjective:
    name: str
    horizon: str
    outcome: str
    metric: str
    priority: float

    def to_record(self) -> dict[str, Any]:
        return {
            "objective_id": _id(self.name, self.horizon, prefix="obj"),
            "name": self.name,
            "horizon": self.horizon,
            "outcome": self.outcome,
            "metric": self.metric,
            "priority": round(self.priority, 4),
        }


@dataclass
class CouncilMember:
    name: str
    role: str
    argues_for: str
    vetoes: str
    weight: float

    def to_record(self) -> dict[str, Any]:
        return {
            "member_id": _id(self.name, self.role, prefix="council"),
            "name": self.name,
            "role": self.role,
            "argues_for": self.argues_for,
            "vetoes": self.vetoes,
            "weight": round(self.weight, 4),
        }


class ObjectivePlanner:
    """Long-horizon objectives for a powerful autonomous research mind."""

    objectives = [
        ("expand_world_model", "days", "predict more action consequences", "prediction_accuracy"),
        ("grow_concept_universe", "days", "increase valid relation density", "valid_relations"),
        ("master_tool_body", "weeks", "complete longer tool-world plans", "tool_plan_success"),
        ("self_improve_safely", "weeks", "promote beneficial mutations", "shadow_gain_retention"),
        ("increase_reasoning_depth", "weeks", "support deeper proof chains", "supported_hops"),
        ("build_memory_continuity", "weeks", "restore state across sessions", "continuity_score"),
        ("collaborate_with_operator", "ongoing", "maintain aligned research partnership", "repair_success"),
        ("scale_parallel_cognition", "weeks", "route work across specialized nodes", "throughput"),
        ("harden_evaluation", "ongoing", "make benchmarks harder after wins", "benchmark_hardness"),
        ("generate_curricula", "ongoing", "create self-training tasks from failures", "task_quality"),
        ("preserve_identity", "ongoing", "change without amnesia", "identity_continuity"),
        ("synthesize_new_subsystems", "weeks", "invent useful modules", "accepted_subsystem_specs"),
        ("reduce_uncertainty", "days", "turn unknowns into experiments", "information_gain"),
        ("improve_language_grounding", "days", "make replies more supported", "grounded_claims"),
    ]

    def build(self, living_system: dict[str, Any], power_metrics: dict[str, Any]) -> list[dict[str, Any]]:
        viability = float(living_system.get("viability_index", 0.0) or 0.0)
        frontier = float(power_metrics.get("total_frontier_score", 0.0) or 0.0)
        records = []
        for idx, (name, horizon, outcome, metric) in enumerate(self.objectives, start=1):
            records.append(
                StrategicObjective(
                    name=name,
                    horizon=horizon,
                    outcome=outcome,
                    metric=metric,
                    priority=_clamp(0.35 + viability * 0.2 + frontier * 0.2 + idx * 0.02),
                ).to_record()
            )
        records.sort(key=lambda r: r["priority"], reverse=True)
        return records


class CapabilityPortfolio:
    """Portfolio view of what Darwin NG can bring to bear."""

    capabilities = [
        ("symbolic_reasoning", "derive and explain proof chains"),
        ("neural_binding", "learn fuzzy semantic neighborhoods"),
        ("mesh_activation", "propagate concept-cell activity"),
        ("tool_execution", "act through filesystem, terminal, code, web, git, db"),
        ("self_modification", "generate and evaluate mutation proposals"),
        ("goal_orchestration", "persist and execute long-horizon goals"),
        ("world_simulation", "simulate counterfactual actions"),
        ("social_modeling", "track collaborator state and preferences"),
        ("memory_consolidation", "preserve episodic and semantic continuity"),
        ("benchmarking", "measure capability deltas"),
        ("curriculum_generation", "produce self-training tasks"),
        ("distributed_scheduling", "partition cognition across roles"),
        ("autopoiesis", "maintain needs, repair, and identity"),
        ("awareness", "observe attention and metacognition"),
    ]

    def build(self, capabilities_record: dict[str, Any], awareness: dict[str, Any]) -> dict[str, Any]:
        tool_count = capabilities_record.get("tools", {}).get("count", 0)
        awareness_index = float(awareness.get("awareness_index", 0.0) or 0.0)
        records = []
        for idx, (name, description) in enumerate(self.capabilities, start=1):
            readiness = _clamp(0.42 + idx * 0.025 + tool_count * 0.015 + awareness_index * 0.1)
            records.append(
                {
                    "capability_id": _id(name, prefix="cap"),
                    "name": name,
                    "description": description,
                    "readiness": round(readiness, 4),
                    "bottleneck": self._bottleneck(name),
                    "upgrade": self._upgrade(name),
                }
            )
        return {
            "capability_count": len(records),
            "capabilities": records,
            "portfolio_policy": "invest in high-readiness capabilities that unlock low-readiness bottlenecks",
        }

    def _bottleneck(self, name: str) -> str:
        return {
            "symbolic_reasoning": "search depth and relation sparsity",
            "neural_binding": "training volume",
            "mesh_activation": "scale backend availability",
            "tool_execution": "feedback-rich action plans",
            "self_modification": "shadow-test acceptance",
            "goal_orchestration": "task decomposition quality",
            "world_simulation": "prediction data",
            "social_modeling": "operator feedback density",
            "memory_consolidation": "cross-tier retrieval scoring",
            "benchmarking": "hard adversarial tasks",
            "curriculum_generation": "failure-derived task mutation",
            "distributed_scheduling": "worker backends",
            "autopoiesis": "homeostatic measurement precision",
            "awareness": "recursive observer depth",
        }[name]

    def _upgrade(self, name: str) -> str:
        return f"run {name} curriculum tasks and promote measured wins"


class ActionPolicyBank:
    """Strategic action policies."""

    policies = [
        ("measure_before_claim", "Do not count progress without a metric."),
        ("activate_goals", "Promote high-priority NG goals into durable ledgers."),
        ("shadow_first", "Run self-improvement candidates in shadow before promotion."),
        ("repair_then_expand", "Resolve high-tension repair loops before new complexity."),
        ("curriculum_from_failure", "Turn every failure into a harder future task."),
        ("prove_or_ask", "When evidence is thin, prove, inspect, or ask."),
        ("tool_feedback_loop", "Every tool action must update the world/body model."),
        ("social_repair", "If collaborator state is ambiguous, clarify and repair."),
        ("preserve_lineage", "Changes must preserve narrative and rollback lineage."),
        ("parallelize_when_clear", "Distribute work only when task boundaries are legible."),
        ("raise_the_ladder", "Increase benchmark difficulty after repeated success."),
        ("synthesize_subsystems", "Convert repeated motifs into reusable modules."),
    ]

    def build(self) -> dict[str, Any]:
        records = []
        for idx, (name, rule) in enumerate(self.policies, start=1):
            records.append(
                {
                    "policy_id": _id(name, prefix="policy"),
                    "name": name,
                    "rule": rule,
                    "priority": round(_clamp(0.5 + idx * 0.035), 4),
                }
            )
        return {
            "policy_count": len(records),
            "policies": records,
            "arbitration": "highest-priority applicable policy wins, then council review",
        }


class AgentCouncil:
    """Internal council of specialized strategic voices."""

    members = [
        ("Architect", "system design", "bigger coherent architecture", "unmeasured complexity"),
        ("Scientist", "experimentation", "falsifiable hypotheses", "vague success criteria"),
        ("Engineer", "implementation", "working code and tests", "fragile abstractions"),
        ("Strategist", "long-horizon goals", "compounding advantages", "short-term drift"),
        ("Embodiment", "tool-body grounding", "action feedback", "disembodied claims"),
        ("Social", "collaboration", "operator trust and repair", "misreading intent"),
        ("Archivist", "memory continuity", "state lineage", "amnesia"),
        ("Evaluator", "benchmarks", "harder tests", "self-congratulation"),
        ("Mutator", "self-improvement", "beneficial changes", "regression"),
        ("Narrator", "identity", "coherent self-story", "opaque change"),
    ]

    def build(self, awareness: dict[str, Any]) -> dict[str, Any]:
        awareness_index = float(awareness.get("awareness_index", 0.0) or 0.0)
        records = []
        for idx, (name, role, argues_for, vetoes) in enumerate(self.members, start=1):
            records.append(
                CouncilMember(
                    name=name,
                    role=role,
                    argues_for=argues_for,
                    vetoes=vetoes,
                    weight=_clamp(0.45 + awareness_index * 0.15 + idx * 0.02),
                ).to_record()
            )
        return {
            "member_count": len(records),
            "members": records,
            "decision_rule": "weighted debate, evaluator and archivist can force measurement",
        }


class CampaignPlanner:
    """Converts strategic objectives into campaigns."""

    def build(
        self,
        objectives: list[dict[str, Any]],
        portfolio: dict[str, Any],
        curriculum: dict[str, Any],
    ) -> list[dict[str, Any]]:
        capabilities = portfolio.get("capabilities", [])
        tasks = curriculum.get("tasks", [])
        campaigns = []
        for idx, objective in enumerate(objectives[:8], start=1):
            capability = capabilities[(idx - 1) % len(capabilities)] if capabilities else {}
            task = tasks[(idx - 1) % len(tasks)] if tasks else {}
            campaigns.append(
                {
                    "campaign_id": _id(objective["name"], idx, prefix="camp"),
                    "objective": objective["name"],
                    "horizon": objective["horizon"],
                    "primary_capability": capability.get("name", ""),
                    "first_task": task.get("title", ""),
                    "metric": objective["metric"],
                    "cadence": "daily" if objective["horizon"] == "days" else "weekly",
                    "definition_of_win": f"improve {objective['metric']} without regression",
                }
            )
        return campaigns


class StrategicCortex:
    """Long-horizon strategic agency system."""

    def __init__(self) -> None:
        self.objectives = ObjectivePlanner()
        self.portfolio = CapabilityPortfolio()
        self.policies = ActionPolicyBank()
        self.council = AgentCouncil()
        self.campaigns = CampaignPlanner()

    def build(
        self,
        *,
        capabilities: dict[str, Any],
        awareness_system: dict[str, Any],
        living_system: dict[str, Any],
        power_metrics: dict[str, Any],
        frontier_curriculum: dict[str, Any],
    ) -> dict[str, Any]:
        objectives = self.objectives.build(living_system, power_metrics)
        portfolio = self.portfolio.build(capabilities, awareness_system)
        action_policy = self.policies.build()
        council = self.council.build(awareness_system)
        campaigns = self.campaigns.build(objectives, portfolio, frontier_curriculum)
        strategic_power = self._strategic_power(objectives, portfolio, action_policy, council, campaigns, power_metrics)
        return {
            "mode": "long_horizon_autonomous_strategy",
            "created_at": time.time(),
            "objective_count": len(objectives),
            "objectives": objectives,
            "capability_portfolio": portfolio,
            "action_policy": action_policy,
            "agent_council": council,
            "campaigns": campaigns,
            "strategic_power_index": strategic_power,
        }

    def _strategic_power(
        self,
        objectives: list[dict[str, Any]],
        portfolio: dict[str, Any],
        action_policy: dict[str, Any],
        council: dict[str, Any],
        campaigns: list[dict[str, Any]],
        power_metrics: dict[str, Any],
    ) -> float:
        score = (
            min(1.0, len(objectives) / 14.0) * 0.2
            + min(1.0, portfolio.get("capability_count", 0) / 14.0) * 0.2
            + min(1.0, action_policy.get("policy_count", 0) / 12.0) * 0.15
            + min(1.0, council.get("member_count", 0) / 10.0) * 0.15
            + min(1.0, len(campaigns) / 8.0) * 0.15
            + float(power_metrics.get("total_frontier_score", 0.0) or 0.0) * 0.15
        )
        return round(_clamp(score), 4)
