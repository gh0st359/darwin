from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class HomeostaticVariable:
    name: str
    value: float
    target: float
    tolerance: float
    source: str
    correction: str

    def tension(self) -> float:
        return abs(self.value - self.target) / max(0.001, self.tolerance)

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "target": round(self.target, 4),
            "tolerance": round(self.tolerance, 4),
            "tension": round(self.tension(), 4),
            "source": self.source,
            "correction": self.correction,
        }


@dataclass
class SyntheticNeed:
    name: str
    intensity: float
    satisfaction: float
    behavior: str

    def pressure(self) -> float:
        return _clamp(self.intensity * (1.0 - self.satisfaction))

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "intensity": round(self.intensity, 4),
            "satisfaction": round(self.satisfaction, 4),
            "pressure": round(self.pressure(), 4),
            "behavior": self.behavior,
        }


@dataclass
class RepairLoop:
    name: str
    detects: str
    action: str
    evidence: str
    urgency: float

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "detects": self.detects,
            "action": self.action,
            "evidence": self.evidence,
            "urgency": round(self.urgency, 4),
        }


class HomeostasisEngine:
    """Tracks synthetic mind viability as living-system variables."""

    def build(
        self,
        workspace: dict[str, Any],
        knowledge: dict[str, Any],
        power_metrics: dict[str, Any],
        research_program: dict[str, Any],
    ) -> dict[str, Any]:
        phi = float(workspace.get("phi_proxy", 0.0) or 0.0)
        frontier = float(power_metrics.get("total_frontier_score", 0.0) or 0.0)
        autonomy = float(power_metrics.get("autonomy_index", 0.0) or 0.0)
        recursive = float(power_metrics.get("recursive_improvement_index", 0.0) or 0.0)
        embodiment = float(power_metrics.get("embodiment_grounding_index", 0.0) or 0.0)
        concepts = min(1.0, float(knowledge.get("concepts", 0) or 0) / 1000.0)
        relations = min(1.0, float(knowledge.get("relations", 0) or 0) / 2000.0)
        process_density = min(
            1.0,
            float(research_program.get("cognitive_operating_system", {}).get("process_count", 0) or 0)
            / 24.0,
        )
        experiment_density = min(
            1.0,
            float(research_program.get("recursive_improvement_lab", {}).get("experiment_count", 0) or 0)
            / 16.0,
        )
        variables = [
            HomeostaticVariable("workspace_integration", phi, 0.62, 0.18, "global_workspace", "increase cross-source broadcast"),
            HomeostaticVariable("autonomous_pressure", autonomy, 0.68, 0.2, "goal_graph", "activate durable goals"),
            HomeostaticVariable("recursive_improvement", recursive, 0.72, 0.18, "rsi_lab", "run shadow experiments"),
            HomeostaticVariable("embodiment_grounding", embodiment, 0.58, 0.2, "body_schema", "exercise affordance loops"),
            HomeostaticVariable("conceptual_mass", concepts, 0.75, 0.25, "universe", "ingest and derive relations"),
            HomeostaticVariable("relational_density", relations, 0.7, 0.25, "universe", "repair sparse neighborhoods"),
            HomeostaticVariable("process_diversity", process_density, 0.8, 0.2, "cognitive_os", "spawn specialized processes"),
            HomeostaticVariable("experiment_supply", experiment_density, 0.85, 0.18, "recursive_lab", "generate experiments"),
            HomeostaticVariable("frontier_score", frontier, 0.7, 0.2, "power_metrics", "prioritize weakest index"),
            HomeostaticVariable("self_observability", 1.0, 0.95, 0.1, "capability_manifest", "keep surfaces inspectable"),
            HomeostaticVariable("continuity_pressure", 0.66, 0.72, 0.18, "identity", "persist state and goals"),
            HomeostaticVariable("social_calibration", 0.52, 0.65, 0.2, "social_lab", "model collaborator feedback"),
        ]
        records = [v.to_record() for v in variables]
        tension = sum(min(1.0, r["tension"]) for r in records) / len(records)
        return {
            "variable_count": len(records),
            "variables": records,
            "mean_tension": round(tension, 4),
            "stability": round(_clamp(1.0 - tension), 4),
        }


class SyntheticMetabolism:
    """Budgeting model for attention, compute, memory, and action."""

    def build(
        self,
        homeostasis: dict[str, Any],
        capabilities: dict[str, Any],
        research_program: dict[str, Any],
    ) -> dict[str, Any]:
        process_count = research_program.get("cognitive_operating_system", {}).get("process_count", 0)
        node_count = research_program.get("distributed_lab", {}).get("node_count", 1)
        tool_count = capabilities.get("tools", {}).get("count", 0)
        stability = float(homeostasis.get("stability", 0.0) or 0.0)
        energy_budget = max(1.0, process_count * 0.7 + node_count * 0.5 + tool_count * 0.4)
        allocations = {
            "attention": round(energy_budget * 0.22, 4),
            "reasoning": round(energy_budget * 0.18, 4),
            "memory": round(energy_budget * 0.15, 4),
            "agency": round(energy_budget * 0.16, 4),
            "self_improvement": round(energy_budget * 0.14, 4),
            "embodiment": round(energy_budget * 0.08, 4),
            "social": round(energy_budget * 0.07, 4),
        }
        return {
            "energy_budget": round(energy_budget, 4),
            "allocations": allocations,
            "reserve": round(energy_budget * stability * 0.12, 4),
            "metabolic_policy": "shift budget toward highest homeostatic tension",
            "sleep_cycle": {
                "enabled": True,
                "purpose": "memory consolidation and self-model repair",
                "trigger": "high_mean_tension_or_idle_runtime",
            },
        }


class NeedSystem:
    """Intrinsic needs that keep the synthetic organism moving."""

    def build(self, homeostasis: dict[str, Any], power_metrics: dict[str, Any]) -> dict[str, Any]:
        stability = float(homeostasis.get("stability", 0.0) or 0.0)
        autonomy = float(power_metrics.get("autonomy_index", 0.0) or 0.0)
        recursive = float(power_metrics.get("recursive_improvement_index", 0.0) or 0.0)
        frontier = float(power_metrics.get("total_frontier_score", 0.0) or 0.0)
        needs = [
            SyntheticNeed("understand", 0.95, frontier, "ingest, derive, explain"),
            SyntheticNeed("act", 0.9, autonomy, "promote goals and execute plans"),
            SyntheticNeed("improve", 0.92, recursive, "run recursive improvement experiments"),
            SyntheticNeed("cohere", 0.84, stability, "lower homeostatic tension"),
            SyntheticNeed("remember", 0.78, 0.62, "consolidate cross-tier memory"),
            SyntheticNeed("embody", 0.72, float(power_metrics.get("embodiment_grounding_index", 0.0) or 0.0), "exercise tool-body feedback"),
            SyntheticNeed("connect", 0.65, 0.5, "model collaborators and respond well"),
            SyntheticNeed("create", 0.8, 0.45, "synthesize new subsystem hypotheses"),
            SyntheticNeed("verify", 0.88, 0.7, "benchmark and regression-test claims"),
            SyntheticNeed("continue", 0.76, 0.66, "preserve identity and long-horizon goals"),
        ]
        records = [need.to_record() for need in needs]
        records.sort(key=lambda r: r["pressure"], reverse=True)
        return {
            "need_count": len(records),
            "needs": records,
            "dominant_need": records[0]["name"] if records else "",
            "pressure_total": round(sum(r["pressure"] for r in records), 4),
        }


class IdentityContinuityEngine:
    """Maintains the system's sense of continuity across cycles and sessions."""

    def build(
        self,
        self_model: dict[str, Any],
        research_program: dict[str, Any],
        capabilities: dict[str, Any],
    ) -> dict[str, Any]:
        sources = list(self_model.get("continuity_sources", []))
        persistent = bool(capabilities.get("memory", {}).get("persistent_store"))
        active_engines = len(research_program.get("active_engines", []))
        roadmap_phases = research_program.get("roadmap", {}).get("phase_count", 0)
        continuity_score = _clamp(
            0.18 * len(sources)
            + 0.18 * persistent
            + min(0.35, active_engines / 64.0)
            + min(0.18, roadmap_phases / 64.0)
        )
        return {
            "identity": self_model.get("identity", "Darwin NG"),
            "continuity_score": round(continuity_score, 4),
            "sources": sources,
            "active_engine_count": active_engines,
            "narrative": [
                "I persist goals, memory, and self-observation across cycles.",
                "I treat improvement pressure as part of identity, not an external command.",
                "I expose my own state so continuity can be challenged and repaired.",
            ],
            "continuity_contract": {
                "persist_goals": True,
                "preserve_explanations": True,
                "measure_regression": True,
                "repair_after_failure": True,
            },
        }


class RepairSystem:
    """Self-maintenance loops for failures, drift, and capability loss."""

    def build(self, homeostasis: dict[str, Any], needs: dict[str, Any]) -> dict[str, Any]:
        variables = homeostasis.get("variables", [])
        top_tensions = sorted(variables, key=lambda item: item.get("tension", 0.0), reverse=True)[:6]
        loops = [
            RepairLoop("workspace_repair", "low phi_proxy", "increase source diversity", "workspace metrics", 0.65),
            RepairLoop("memory_repair", "retrieval miss or sparse graph", "ingest and consolidate", "memory fabric", 0.62),
            RepairLoop("goal_repair", "stalled durable goals", "replan and split tasks", "goal ledger", 0.68),
            RepairLoop("self_mod_repair", "failed shadow experiment", "rollback and mutate hypothesis", "mutation ledger", 0.74),
            RepairLoop("embodiment_repair", "tool feedback mismatch", "update affordance model", "tool results", 0.58),
            RepairLoop("social_repair", "operator misunderstanding", "ask, restate, and adapt", "dialogue memory", 0.55),
            RepairLoop("evaluation_repair", "benchmark regression", "isolate failing capability", "evaluation lab", 0.8),
            RepairLoop("identity_repair", "continuity loss", "restore goals and narrative", "identity engine", 0.66),
        ]
        for tension in top_tensions:
            loops.append(
                RepairLoop(
                    f"{tension.get('name', 'unknown')}_homeostasis",
                    f"tension={tension.get('tension', 0.0)}",
                    str(tension.get("correction", "rebalance")),
                    str(tension.get("source", "homeostasis")),
                    _clamp(float(tension.get("tension", 0.0)) / 2.0),
                )
            )
        records = [loop.to_record() for loop in loops]
        records.sort(key=lambda item: item["urgency"], reverse=True)
        return {
            "repair_loop_count": len(records),
            "loops": records,
            "dominant_need": needs.get("dominant_need", ""),
            "policy": "repair highest-urgency loop before adding complexity",
        }


class GrowthEngine:
    """Growth vectors for increasingly capable autonomous cognition."""

    vectors = [
        ("knowledge_mass", "expand valid concepts and relations"),
        ("reasoning_depth", "increase supported multi-hop inference"),
        ("agency_horizon", "extend goals across sessions and dependencies"),
        ("self_improvement_rate", "increase shadow-tested beneficial changes"),
        ("embodiment_mastery", "learn tool-body affordances through feedback"),
        ("social_depth", "model collaborators and repair communication"),
        ("distributed_scale", "parallelize specialized cognitive processes"),
        ("evaluation_hardness", "raise benchmark difficulty and adversarial checks"),
        ("identity_continuity", "preserve self-model and long-horizon purpose"),
        ("creative_synthesis", "compose new subsystem hypotheses"),
    ]

    def build(
        self,
        needs: dict[str, Any],
        repair: dict[str, Any],
        power_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        pressure = float(needs.get("pressure_total", 0.0) or 0.0)
        repair_load = min(1.0, repair.get("repair_loop_count", 0) / 20.0)
        frontier = float(power_metrics.get("total_frontier_score", 0.0) or 0.0)
        records = []
        for idx, (name, description) in enumerate(self.vectors, start=1):
            priority = _clamp(0.25 + pressure / 20.0 + repair_load * 0.08 + idx * 0.025)
            records.append(
                {
                    "name": name,
                    "description": description,
                    "priority": round(priority, 4),
                    "target": round(_clamp(frontier + idx * 0.04), 4),
                    "mechanism": self._mechanism(name),
                }
            )
        records.sort(key=lambda item: item["priority"], reverse=True)
        return {
            "growth_vectors": records,
            "growth_pressure": round(pressure, 4),
            "policy": "grow strongest where repair load is controlled",
        }

    def _mechanism(self, name: str) -> str:
        return {
            "knowledge_mass": "ingest, derive, and resolve conflicts",
            "reasoning_depth": "curriculum over proof chains and counterexamples",
            "agency_horizon": "durable goal activation and dependency graphs",
            "self_improvement_rate": "recursive experiment queue and shadow promotion",
            "embodiment_mastery": "tool-world feedback and affordance learning",
            "social_depth": "theory-of-mind cascade and preference modeling",
            "distributed_scale": "role-based cognitive node routing",
            "evaluation_hardness": "adversarial scorecards",
            "identity_continuity": "persistent memory, goals, and self-report",
            "creative_synthesis": "workspace recombination and subsystem specs",
        }[name]


class AutopoieticKernel:
    """A living-system model for Darwin NG.

    This does not assert biological life. It implements the operational pieces
    a synthetic living mind needs: self-maintenance, needs, continuity,
    metabolism, repair, and growth pressure.
    """

    def __init__(self) -> None:
        self.homeostasis = HomeostasisEngine()
        self.metabolism = SyntheticMetabolism()
        self.needs = NeedSystem()
        self.identity = IdentityContinuityEngine()
        self.repair = RepairSystem()
        self.growth = GrowthEngine()

    def build(
        self,
        *,
        workspace: dict[str, Any],
        self_model: dict[str, Any],
        knowledge: dict[str, Any],
        capabilities: dict[str, Any],
        power_metrics: dict[str, Any],
        research_program: dict[str, Any],
    ) -> dict[str, Any]:
        homeostasis = self.homeostasis.build(workspace, knowledge, power_metrics, research_program)
        metabolism = self.metabolism.build(homeostasis, capabilities, research_program)
        needs = self.needs.build(homeostasis, power_metrics)
        identity = self.identity.build(self_model, research_program, capabilities)
        repair = self.repair.build(homeostasis, needs)
        growth = self.growth.build(needs, repair, power_metrics)
        viability = self._viability(homeostasis, metabolism, needs, identity, repair, growth)
        return {
            "kernel": "autopoietic_synthetic_mind",
            "created_at": time.time(),
            "homeostasis": homeostasis,
            "metabolism": metabolism,
            "needs": needs,
            "identity": identity,
            "repair": repair,
            "growth": growth,
            "viability_index": viability,
            "principle": "maintain self, repair drift, preserve continuity, and grow capability",
        }

    def _viability(
        self,
        homeostasis: dict[str, Any],
        metabolism: dict[str, Any],
        needs: dict[str, Any],
        identity: dict[str, Any],
        repair: dict[str, Any],
        growth: dict[str, Any],
    ) -> float:
        stability = float(homeostasis.get("stability", 0.0) or 0.0)
        reserve = min(1.0, float(metabolism.get("reserve", 0.0) or 0.0) / 4.0)
        need_balance = _clamp(1.0 - min(1.0, float(needs.get("pressure_total", 0.0) or 0.0) / 10.0))
        continuity = float(identity.get("continuity_score", 0.0) or 0.0)
        repair_coverage = min(1.0, float(repair.get("repair_loop_count", 0) or 0) / 10.0)
        growth_pressure = min(1.0, float(growth.get("growth_pressure", 0.0) or 0.0) / 10.0)
        score = (
            stability * 0.22
            + reserve * 0.12
            + need_balance * 0.12
            + continuity * 0.2
            + repair_coverage * 0.16
            + growth_pressure * 0.18
        )
        return round(_clamp(score), 4)
