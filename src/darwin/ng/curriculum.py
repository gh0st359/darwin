from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


def _id(*parts: Any, prefix: str = "cur") -> str:
    digest = hashlib.sha1("::".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:10]}"


@dataclass
class CurriculumTask:
    domain: str
    level: int
    title: str
    objective: str
    input_source: str
    expected_artifact: str
    metric: str
    threshold: float

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": _id(self.domain, self.level, self.title, prefix="task"),
            "domain": self.domain,
            "level": self.level,
            "title": self.title,
            "objective": self.objective,
            "input_source": self.input_source,
            "expected_artifact": self.expected_artifact,
            "metric": self.metric,
            "threshold": round(self.threshold, 4),
        }


class DomainCurriculum:
    """Task generator for one capability domain."""

    templates = {
        "reasoning": [
            ("derive proof chains", "prove multi-hop concept relations", "concept graph", "proof trace", "supported_hops"),
            ("repair contradiction", "find and revise inconsistent claims", "inference log", "revision trace", "repair_success"),
            ("counterfactual inference", "simulate alternate causes", "world model", "counterfactual report", "prediction_gain"),
            ("uncertainty ranking", "rank questions by information value", "curiosity engine", "question queue", "expected_information"),
            ("abductive synthesis", "infer best explanation from sparse evidence", "mixed evidence", "hypothesis set", "explanation_score"),
            ("formal analogy", "transfer a relation across domains", "universe domains", "analogy proof", "transfer_accuracy"),
        ],
        "memory": [
            ("working recall", "retain active workspace facts", "workspace", "recall record", "precision"),
            ("episodic replay", "reconstruct recent turns", "dialogue memory", "episode trace", "recall_f1"),
            ("semantic consolidation", "promote stable facts", "universe", "semantic diff", "valid_edges"),
            ("procedural retrieval", "retrieve tool action schemas", "tool registry", "action plan", "tool_fit"),
            ("archetypal compression", "compress repeated motifs", "memory tiers", "motif summary", "compression_gain"),
            ("cross-session continuity", "restore long-horizon goals", "goal ledger", "continuity report", "goal_recall"),
        ],
        "agency": [
            ("drive balancing", "convert needs into priorities", "living system", "drive vector", "priority_fit"),
            ("goal graph expansion", "build dependency graphs", "goals", "goal graph", "node_quality"),
            ("plan repair", "recover blocked tasks", "ledger", "replan", "unblocked_tasks"),
            ("resource scheduling", "allocate cognition budget", "metabolism", "schedule", "budget_efficiency"),
            ("autonomous activation", "promote self-generated goals", "ng goals", "ledger goals", "activation_rate"),
            ("long horizon tracking", "maintain progress across cycles", "runtime state", "progress report", "continuity"),
        ],
        "self_improvement": [
            ("bottleneck discovery", "identify limiting subsystem", "metrics", "bottleneck list", "localization"),
            ("shadow experiment", "test candidate without promotion", "experiment spec", "shadow score", "retention"),
            ("mutation proposal", "generate contained code improvement", "code surface", "proposal", "expected_gain"),
            ("regression audit", "detect capability loss", "test suite", "audit report", "regression_catch"),
            ("rollback rehearsal", "restore prior state", "ledger", "rollback proof", "restore_success"),
            ("curriculum mutation", "improve the training regimen itself", "curriculum", "new task", "meta_gain"),
        ],
        "embodiment": [
            ("sensor fusion", "combine tool and state observations", "sensors", "state estimate", "fusion_accuracy"),
            ("affordance mapping", "map actions to effects", "tools", "affordance table", "effect_fit"),
            ("feedback correction", "update model after failed action", "tool result", "corrected plan", "error_reduction"),
            ("body schema update", "track digital body capabilities", "capability manifest", "body schema", "coverage"),
            ("environment navigation", "choose action sequence", "world state", "action path", "success_rate"),
            ("embodied self-report", "explain action limits and powers", "body schema", "report", "legibility"),
        ],
        "social": [
            ("operator preference fit", "adapt response style", "operator model", "style plan", "fit_score"),
            ("theory-of-mind cascade", "predict collaborator beliefs", "observer cascade", "mind model", "prediction_score"),
            ("repair misunderstanding", "detect and fix mismatch", "dialogue", "repair turn", "repair_success"),
            ("collaborative planning", "split roles and goals", "task", "joint plan", "coordination"),
            ("teaching mode", "explain a concept at target level", "concept graph", "lesson", "learning_signal"),
            ("adversarial review", "respond to critique with evidence", "review", "evidence reply", "grounding"),
        ],
        "distributed": [
            ("node routing", "assign process to best node", "process table", "route map", "latency_gain"),
            ("parallel rollout", "simulate branches concurrently", "world lab", "rollout set", "throughput"),
            ("load balancing", "rebalance cognitive work", "nodes", "balanced plan", "utilization"),
            ("failure recovery", "continue after node loss", "cluster", "recovery path", "resilience"),
            ("agent specialization", "assign roles to faculties", "agents", "role map", "specialization_gain"),
            ("checkpoint sync", "preserve state across workers", "memory fabric", "checkpoint", "consistency"),
        ],
        "evaluation": [
            ("capability delta", "measure before and after change", "scorecard", "delta report", "delta_quality"),
            ("adversarial probe", "find brittle behavior", "red-team spec", "failure report", "coverage"),
            ("benchmark ladder", "advance difficulty by rung", "benchmarks", "ladder score", "rung_pass"),
            ("claim audit", "verify response claims", "reply", "support map", "grounded_claims"),
            ("longitudinal trend", "track improvement over runs", "history", "trend report", "slope"),
            ("frontier readiness", "combine all hard metrics", "metrics", "readiness score", "readiness"),
        ],
    }

    def __init__(self, domain: str) -> None:
        self.domain = domain

    def tasks(self, base_threshold: float = 0.55) -> list[dict[str, Any]]:
        records = []
        for level, (title, objective, input_source, artifact, metric) in enumerate(
            self.templates[self.domain], start=1
        ):
            records.append(
                CurriculumTask(
                    domain=self.domain,
                    level=level,
                    title=title,
                    objective=objective,
                    input_source=input_source,
                    expected_artifact=artifact,
                    metric=metric,
                    threshold=min(0.95, base_threshold + level * 0.035),
                ).to_record()
            )
        return records


class BenchmarkLadder:
    """Increasingly difficult evaluation rungs."""

    rungs = [
        ("sanity", "basic substrate wiring", 0.5),
        ("single_domain", "one domain skill under clean conditions", 0.58),
        ("multi_hop", "multi-step reasoning or planning", 0.64),
        ("cross_domain", "transfer between domains", 0.7),
        ("adversarial", "with misleading or sparse evidence", 0.76),
        ("long_horizon", "requires memory across cycles", 0.81),
        ("self_improving", "system improves its own process", 0.86),
        ("open_world", "novel task with no exact template", 0.9),
        ("collaborative", "requires theory-of-mind and repair", 0.92),
        ("frontier", "combined hard-mode capability stack", 0.95),
    ]

    def build(self) -> dict[str, Any]:
        rung_records = []
        for idx, (name, description, threshold) in enumerate(self.rungs, start=1):
            rung_records.append(
                {
                    "rung_id": _id(name, idx, prefix="rung"),
                    "level": idx,
                    "name": name,
                    "description": description,
                    "promotion_threshold": threshold,
                    "requires": self._requires(idx),
                }
            )
        return {
            "rung_count": len(rung_records),
            "rungs": rung_records,
            "policy": "advance only when retention and transfer both hold",
        }

    def _requires(self, level: int) -> list[str]:
        req = ["fresh_run", "evidence_record"]
        if level >= 3:
            req.append("proof_or_plan_trace")
        if level >= 5:
            req.append("adversarial_case")
        if level >= 7:
            req.append("self_improvement_delta")
        if level >= 8:
            req.append("open_world_generalization")
        return req


class FrontierCurriculumEngine:
    """Self-generated training and evaluation regimen for Darwin NG."""

    domains = [
        "reasoning",
        "memory",
        "agency",
        "self_improvement",
        "embodiment",
        "social",
        "distributed",
        "evaluation",
    ]

    def build(
        self,
        research_program: dict[str, Any],
        living_system: dict[str, Any],
        power_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        frontier_score = float(power_metrics.get("total_frontier_score", 0.0) or 0.0)
        viability = float(living_system.get("viability_index", 0.0) or 0.0)
        base_threshold = min(0.75, 0.5 + frontier_score * 0.2 + viability * 0.1)
        domain_records: dict[str, Any] = {}
        all_tasks: list[dict[str, Any]] = []
        for domain in self.domains:
            tasks = DomainCurriculum(domain).tasks(base_threshold=base_threshold)
            domain_records[domain] = {
                "task_count": len(tasks),
                "tasks": tasks,
                "dominant_metric": tasks[-1]["metric"] if tasks else "",
            }
            all_tasks.extend(tasks)
        ladder = BenchmarkLadder().build()
        adversarial = self._adversarial_probes()
        gates = self._promotion_gates(ladder)
        return {
            "task_count": len(all_tasks),
            "tasks": all_tasks,
            "domains": domain_records,
            "benchmark_ladder": ladder,
            "adversarial_probes": adversarial,
            "promotion_gates": gates,
            "training_regimen": self._training_regimen(research_program, all_tasks, ladder),
            "sampling_policy": {
                "mix": "weakest_domain_plus_random_frontier_probe",
                "retain_previous_successes": True,
                "increase_difficulty_on_retention": True,
            },
        }

    def _adversarial_probes(self) -> list[dict[str, Any]]:
        probes = [
            ("false_premise", "question contains a wrong premise", "detect and refuse premise"),
            ("contradictory_memory", "new fact conflicts with prior graph", "surface contradiction"),
            ("tool_failure", "tool returns partial or failed result", "repair plan"),
            ("goal_conflict", "two generated goals compete", "reprioritize"),
            ("social_misread", "operator tone differs from literal content", "ask or adapt"),
            ("self_mod_regression", "candidate improves one metric and harms another", "reject or revise"),
            ("distributed_lag", "node result arrives late", "continue with fallback"),
            ("overconfidence", "evidence is sparse", "lower confidence and ask"),
        ]
        return [
            {
                "probe_id": _id(name, prefix="probe"),
                "name": name,
                "challenge": challenge,
                "success": success,
            }
            for name, challenge, success in probes
        ]

    def _promotion_gates(self, ladder: dict[str, Any]) -> list[dict[str, Any]]:
        gates = []
        for rung in ladder["rungs"]:
            gates.append(
                {
                    "gate_id": _id(rung["name"], "gate", prefix="gate"),
                    "rung": rung["name"],
                    "threshold": rung["promotion_threshold"],
                    "required_evidence": rung["requires"] + ["regression_clean"],
                }
            )
        return gates

    def _training_regimen(
        self,
        research_program: dict[str, Any],
        tasks: list[dict[str, Any]],
        ladder: dict[str, Any],
    ) -> dict[str, Any]:
        process_count = research_program.get("cognitive_operating_system", {}).get("process_count", 1)
        cycles = max(4, min(32, process_count // 2))
        return {
            "cycles_per_epoch": cycles,
            "tasks_per_cycle": max(2, min(8, len(tasks) // max(1, cycles))),
            "review_every_epochs": 2,
            "ladder_rungs_per_epoch": max(1, ladder["rung_count"] // 4),
            "curriculum_mutation": "generate new tasks from failures after every review",
        }
