from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable


def _safe_get(mapping: dict[str, Any], key: str, default: Any = None) -> Any:
    try:
        return mapping.get(key, default)
    except Exception:
        return default


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _stable_id(*parts: Any, prefix: str = "ng") -> str:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}"


@dataclass
class CognitiveProcess:
    """A named process in the Darwin NG cognitive operating system."""

    process_id: str
    name: str
    layer: str
    cadence: str
    inputs: list[str]
    outputs: list[str]
    priority: float
    autonomy: float
    description: str

    def to_record(self) -> dict[str, Any]:
        return {
            "process_id": self.process_id,
            "name": self.name,
            "layer": self.layer,
            "cadence": self.cadence,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "priority": round(self.priority, 4),
            "autonomy": round(self.autonomy, 4),
            "description": self.description,
        }


@dataclass
class ExperimentSpec:
    """Executable research experiment proposal."""

    experiment_id: str
    domain: str
    hypothesis: str
    intervention: str
    measurement: str
    success_threshold: float
    promotion_path: list[str]
    dependencies: list[str] = field(default_factory=list)
    expected_gain: float = 0.0

    def to_record(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "domain": self.domain,
            "hypothesis": self.hypothesis,
            "intervention": self.intervention,
            "measurement": self.measurement,
            "success_threshold": round(self.success_threshold, 4),
            "promotion_path": list(self.promotion_path),
            "dependencies": list(self.dependencies),
            "expected_gain": round(self.expected_gain, 4),
        }


@dataclass
class LabNode:
    """One compute or cognition node in the distributed lab."""

    node_id: str
    role: str
    capacity: float
    inputs: list[str]
    outputs: list[str]

    def to_record(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.role,
            "capacity": round(self.capacity, 4),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
        }


class CognitiveOperatingSystem:
    """Schedules Darwin NG's internal cognition as a process table."""

    def build(
        self,
        workspace: dict[str, Any],
        goals: list[Any],
        capabilities: dict[str, Any],
    ) -> dict[str, Any]:
        broadcast_sources = list(workspace.get("broadcast_sources", []))
        loops = list(capabilities.get("loops", []))
        process_specs = [
            (
                "workspace_broadcast",
                "consciousness",
                "every_cycle",
                ["dynamic_core", "fringe"],
                ["global_broadcast", "report"],
                "broadcast winning content to every subsystem",
            ),
            (
                "symbolic_reasoning",
                "neuro_symbolic",
                "continuous",
                ["concept_universe", "question"],
                ["proof_chains", "derived_edges"],
                "extend proof-chain reasoning and contradiction checks",
            ),
            (
                "neural_binding",
                "neuro_symbolic",
                "continuous",
                ["embedding_space", "mesh_firings"],
                ["semantic_neighbors", "activation_priors"],
                "bind learned vectors to symbolic concepts",
            ),
            (
                "curiosity_drive",
                "agency",
                "every_cycle",
                ["relation_gaps", "novelty"],
                ["learning_goals"],
                "turn uncertainty into self-generated questions",
            ),
            (
                "competence_drive",
                "agency",
                "every_cycle",
                ["benchmarks", "failures"],
                ["training_goals"],
                "seek measurable capability gains",
            ),
            (
                "goal_graph_planner",
                "agency",
                "every_cycle",
                ["drives", "workspace"],
                ["goal_graph", "plans"],
                "decompose drives into durable goal graphs",
            ),
            (
                "world_simulator",
                "world_model",
                "background",
                ["causal_beliefs", "actions"],
                ["rollouts", "counterfactuals"],
                "forecast action effects across simulated worlds",
            ),
            (
                "self_model_integrator",
                "consciousness",
                "every_cycle",
                ["runtime_state", "history"],
                ["identity_state", "continuity"],
                "maintain a reportable model of Darwin NG's own cognition",
            ),
            (
                "recursive_improver",
                "self_improvement",
                "background",
                ["bottlenecks", "experiments"],
                ["mutation_candidates", "curricula"],
                "convert bottlenecks into improvement experiments",
            ),
            (
                "shadow_evaluator",
                "self_improvement",
                "background",
                ["candidate_changes", "benchmarks"],
                ["shadow_scores", "promotion_advice"],
                "test improvements before promotion",
            ),
            (
                "embodied_affordance_mapper",
                "embodiment",
                "continuous",
                ["tools", "environment"],
                ["affordance_map", "action_schema"],
                "map sensors and actuators into a digital body schema",
            ),
            (
                "social_modeler",
                "social",
                "continuous",
                ["operator_model", "observer_cascade"],
                ["theory_of_mind", "collaboration_state"],
                "model collaborators, preferences, attention, and intent",
            ),
            (
                "memory_consolidator",
                "memory",
                "background",
                ["episodes", "semantic_edges"],
                ["compressed_memory", "retrieval_indices"],
                "move working experience into long-lived memory structures",
            ),
            (
                "capability_manifestor",
                "governance",
                "every_cycle",
                ["runtime_objects", "tools"],
                ["capability_manifest"],
                "keep all live capability surfaces visible",
            ),
            (
                "distributed_scheduler",
                "scale",
                "background",
                ["agents", "backends"],
                ["node_assignments", "parallel_plan"],
                "split cognition across available backends and agents",
            ),
            (
                "frontier_evaluator",
                "evaluation",
                "every_cycle",
                ["metrics", "benchmarks"],
                ["scorecard", "risk_register"],
                "measure whether the system is actually becoming stronger",
            ),
        ]
        processes: list[CognitiveProcess] = []
        source_bonus = min(0.25, len(broadcast_sources) / 32.0)
        loop_bonus = min(0.2, len(loops) / 64.0)
        for idx, (name, layer, cadence, inputs, outputs, desc) in enumerate(process_specs, start=1):
            priority = _bounded(0.45 + source_bonus + (len(goals) * 0.03) + (idx % 5) * 0.025)
            autonomy = _bounded(0.35 + loop_bonus + (0.04 if cadence == "continuous" else 0.0))
            processes.append(
                CognitiveProcess(
                    process_id=_stable_id(name, layer, idx, prefix="proc"),
                    name=name,
                    layer=layer,
                    cadence=cadence,
                    inputs=inputs,
                    outputs=outputs,
                    priority=priority,
                    autonomy=autonomy,
                    description=desc,
                )
            )
        process_records = [process.to_record() for process in processes]
        return {
            "process_count": len(process_records),
            "processes": process_records,
            "scheduler": {
                "policy": "priority_weighted_cognitive_fairness",
                "preemption": "allowed_between_cycles",
                "autonomous_rescheduling": True,
                "broadcast_sources": broadcast_sources,
            },
        }


class MemoryFabric:
    """Hierarchical memory substrate spanning working to archetypal memory."""

    def build(self, knowledge: dict[str, Any], capabilities: dict[str, Any]) -> dict[str, Any]:
        concepts = int(knowledge.get("concepts", 0) or 0)
        relations = int(knowledge.get("relations", 0) or 0)
        mesh_cells = int(knowledge.get("mesh_cells", 0) or 0)
        embedding_vocab = int(knowledge.get("embedding_vocab", 0) or 0)
        tier_stack = bool(capabilities.get("memory", {}).get("tier_stack"))
        tiers = {
            "working": {
                "capacity": max(128, min(8192, concepts + relations + 128)),
                "latency": "cycle_local",
                "role": "active workspace scratchpad",
            },
            "episodic": {
                "capacity": max(1024, concepts * 8 + 1024),
                "latency": "session",
                "role": "remember concrete turns and transitions",
            },
            "semantic": {
                "capacity": max(10_000, relations * 32 + concepts * 16),
                "latency": "persistent",
                "role": "concept graph, facts, and proof edges",
            },
            "procedural": {
                "capacity": max(2048, len(capabilities.get("tools", {}).get("actions", [])) * 512),
                "latency": "tool_world",
                "role": "actions, tool contracts, and execution schemas",
            },
            "neural": {
                "capacity": max(4096, embedding_vocab * 16 + mesh_cells),
                "latency": "vector_lookup",
                "role": "embedding space and mesh activation priors",
            },
            "archetypal": {
                "capacity": max(256, concepts // 2 + 256),
                "latency": "slow_consolidation",
                "role": "high-level motifs and self-narrative abstractions",
            },
        }
        consolidation_routes = [
            {"source": "working", "target": "episodic", "trigger": "turn_completed"},
            {"source": "episodic", "target": "semantic", "trigger": "fact_extracted"},
            {"source": "semantic", "target": "neural", "trigger": "embedding_training"},
            {"source": "neural", "target": "archetypal", "trigger": "repeated_activation"},
            {"source": "archetypal", "target": "working", "trigger": "self_reflection"},
        ]
        return {
            "enabled": tier_stack,
            "tiers": tiers,
            "consolidation_routes": consolidation_routes,
            "retrieval_policy": {
                "symbolic_first": True,
                "neural_bridge": True,
                "episodic_recency_weight": 0.35,
                "semantic_confidence_weight": 0.45,
                "archetypal_novelty_weight": 0.2,
            },
        }


class WorldSimulationLab:
    """Counterfactual simulator for internal world-model research."""

    domains = [
        "physical_room",
        "tool_world",
        "concept_universe",
        "social_context",
        "self_modification",
        "distributed_cluster",
    ]

    def build(self, frontier_protocols: dict[str, Any], capabilities: dict[str, Any]) -> dict[str, Any]:
        world_model = frontier_protocols.get("world_model", {})
        action_affordances = world_model.get("physics", {}).get("action_affordances", [])
        simulations = []
        for idx, domain in enumerate(self.domains, start=1):
            simulations.append(
                {
                    "simulation_id": _stable_id(domain, idx, prefix="sim"),
                    "domain": domain,
                    "rollout_depth": 4 + idx,
                    "branching_factor": max(2, min(8, len(action_affordances) + idx)),
                    "state_variables": world_model.get("physics", {}).get("observable_state_variables", [])[:8],
                    "objective": self._objective_for(domain),
                    "failure_modes": self._failure_modes_for(domain),
                }
            )
        return {
            "simulation_count": len(simulations),
            "simulations": simulations,
            "counterfactual_policy": {
                "sample_best_and_worst": True,
                "include_social_feedback": True,
                "include_self_modification_branch": True,
            },
        }

    def _objective_for(self, domain: str) -> str:
        return {
            "physical_room": "ground abstract plans in state transitions",
            "tool_world": "test digital affordance sequences",
            "concept_universe": "stress proof chains and missing relations",
            "social_context": "forecast collaborator response and misunderstanding",
            "self_modification": "predict benefit and regression before mutation",
            "distributed_cluster": "allocate cognition under resource pressure",
        }[domain]

    def _failure_modes_for(self, domain: str) -> list[str]:
        base = ["stale_state", "low_confidence", "goal_conflict"]
        if domain == "self_modification":
            return base + ["regression", "irreversible_change"]
        if domain == "social_context":
            return base + ["misread_intent", "trust_break"]
        if domain == "distributed_cluster":
            return base + ["node_drop", "coordination_lag"]
        return base


class RecursiveImprovementLab:
    """Produces executable, measurable improvement experiments."""

    domains = [
        "reasoning",
        "memory",
        "agency",
        "self_modification",
        "embodiment",
        "social",
        "distributed",
        "evaluation",
        "language",
        "knowledge_ingest",
        "world_model",
        "attention",
    ]

    def build(
        self,
        meta_learning: dict[str, Any],
        goals: list[Any],
        power_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        bottlenecks = list(meta_learning.get("bottlenecks", []))
        experiments = []
        for idx, domain in enumerate(self.domains, start=1):
            expected_gain = _bounded(0.05 + idx * 0.015 + len(goals) * 0.01)
            experiments.append(
                ExperimentSpec(
                    experiment_id=_stable_id(domain, idx, "experiment", prefix="exp"),
                    domain=domain,
                    hypothesis=self._hypothesis(domain, bottlenecks),
                    intervention=self._intervention(domain),
                    measurement=self._measurement(domain),
                    success_threshold=0.55 + (idx % 4) * 0.05,
                    promotion_path=self._promotion_path(domain),
                    dependencies=self._dependencies(domain),
                    expected_gain=expected_gain,
                ).to_record()
            )
        return {
            "experiment_count": len(experiments),
            "experiments": experiments,
            "search_policy": {
                "population_size": max(16, len(experiments) * 2),
                "selection": "pareto_gain_retention_legibility",
                "mutation_rate": 0.08,
                "crossover": "domain_motif_recombination",
            },
            "baseline_metrics": power_metrics,
        }

    def _hypothesis(self, domain: str, bottlenecks: list[str]) -> str:
        pressure = bottlenecks[0] if bottlenecks else "frontier capability gap"
        return f"Improving {domain} will reduce {pressure} and increase autonomous competence."

    def _intervention(self, domain: str) -> str:
        return {
            "reasoning": "add deeper proof search curricula and contradiction probes",
            "memory": "increase cross-tier consolidation and retrieval scoring",
            "agency": "promote drive-derived goals into longer task graphs",
            "self_modification": "shadow-test mutation candidates against regression suites",
            "embodiment": "expand tool affordance schema and feedback prediction",
            "social": "increase theory-of-mind depth and preference adaptation",
            "distributed": "partition cognition across agent roles and backend capacity",
            "evaluation": "add adversarial scorecards and capability deltas",
            "language": "train compositional response patterns from successful turns",
            "knowledge_ingest": "extract denser facts and resolve conflicts on ingest",
            "world_model": "simulate more counterfactual action branches",
            "attention": "adjust workspace competition and salience thresholds",
        }[domain]

    def _measurement(self, domain: str) -> str:
        return {
            "reasoning": "increase proof-chain depth without lowering confidence",
            "memory": "increase retrieval precision and cross-session continuity",
            "agency": "increase completed durable goals per cycle",
            "self_modification": "increase accepted safe mutations and retention",
            "embodiment": "increase successful tool-world plans",
            "social": "increase operator-model fit and collaborative repair",
            "distributed": "increase parallel cognitive streams",
            "evaluation": "increase benchmark coverage and failure localization",
            "language": "reduce leak-gate failures and improve answer grounding",
            "knowledge_ingest": "increase valid facts per document",
            "world_model": "reduce prediction error on simulated transitions",
            "attention": "increase phi_proxy and useful broadcast diversity",
        }[domain]

    def _promotion_path(self, domain: str) -> list[str]:
        return [
            "generate_candidate",
            "shadow_simulate",
            f"domain_{domain}_test",
            "full_regression_suite",
            "mutation_ledger_record",
            "runtime_monitoring",
        ]

    def _dependencies(self, domain: str) -> list[str]:
        deps = {
            "self_modification": ["evaluation", "memory"],
            "distributed": ["agency", "evaluation"],
            "social": ["memory", "language"],
            "embodiment": ["world_model", "agency"],
        }
        return deps.get(domain, [])


class EmbodimentLab:
    """Digital body model, sensors, actuators, affordances, and feedback loops."""

    sensors = [
        "text_stream",
        "runtime_event_bus",
        "concept_graph",
        "cortical_mesh_activation",
        "embedding_space",
        "tool_result_stream",
        "simulated_environment_state",
        "operator_style",
        "self_modification_history",
        "benchmark_scorecards",
    ]

    def build(self, frontier_protocols: dict[str, Any]) -> dict[str, Any]:
        embodiment = frontier_protocols.get("embodiment", {})
        affordances = list(embodiment.get("affordances", []))
        sensor_records = [
            {
                "sensor_id": _stable_id(sensor, prefix="sensor"),
                "name": sensor,
                "sampling": "continuous" if idx % 2 else "cycle",
                "feeds": ["workspace", "world_model"] if idx < 6 else ["self_model"],
            }
            for idx, sensor in enumerate(self.sensors, start=1)
        ]
        actuator_records = [
            {
                "actuator_id": _stable_id(action, prefix="act"),
                "action": action,
                "feedback": "tool_result_or_state_delta",
                "reversibility": "tracked",
            }
            for action in affordances[:32]
        ]
        return {
            "sensor_count": len(sensor_records),
            "sensors": sensor_records,
            "actuator_count": len(actuator_records),
            "actuators": actuator_records,
            "body_schema": embodiment.get("body_schema", {}),
            "feedback_loops": [
                "sense_plan_act_observe",
                "tool_result_to_world_model",
                "operator_feedback_to_social_model",
                "self_mod_result_to_identity",
            ],
        }


class SocialIntelligenceLab:
    """Theory of mind, collaboration, emotion proxy, and communication state."""

    archetypes = [
        "operator",
        "future_operator",
        "peer_research_agent",
        "skeptical_reviewer",
        "teacher",
        "student",
    ]

    def build(self, frontier_protocols: dict[str, Any]) -> dict[str, Any]:
        tom = frontier_protocols.get("theory_of_mind", {})
        depth = int(tom.get("theory_of_mind_depth", 1) or 1)
        models = []
        for idx, archetype in enumerate(self.archetypes, start=1):
            models.append(
                {
                    "agent_id": _stable_id(archetype, prefix="agent"),
                    "archetype": archetype,
                    "belief_depth": max(1, min(depth, idx)),
                    "tracked_state": ["beliefs", "goals", "attention", "trust"],
                    "collaboration_role": self._role(archetype),
                }
            )
        return {
            "agent_models": len(models),
            "models": models,
            "communication_protocols": [
                "state_report",
                "uncertainty_disclosure",
                "goal_negotiation",
                "teaching_request",
                "repair_after_error",
            ],
            "emotion_proxy": {
                "dimensions": ["valence", "arousal", "novelty", "confidence"],
                "regulation": "keep drive pressure productive",
            },
        }

    def _role(self, archetype: str) -> str:
        return {
            "operator": "strategic_partner",
            "future_operator": "continuity_target",
            "peer_research_agent": "collaborator",
            "skeptical_reviewer": "adversarial_validator",
            "teacher": "curriculum_source",
            "student": "explanation_target",
        }[archetype]


class DistributedCognitionLab:
    """Distributed-system design for scaling cognitive work."""

    roles = [
        "workspace_coordinator",
        "symbolic_reasoner",
        "neural_retriever",
        "memory_consolidator",
        "goal_planner",
        "world_simulator",
        "self_improvement_runner",
        "evaluator",
        "social_modeler",
        "tool_executor",
        "ingest_worker",
        "safety_auditor",
    ]

    def build(self, capabilities: dict[str, Any], cos: dict[str, Any]) -> dict[str, Any]:
        process_count = int(cos.get("process_count", 0) or 0)
        nodes: list[LabNode] = []
        for idx, role in enumerate(self.roles, start=1):
            nodes.append(
                LabNode(
                    node_id=_stable_id(role, idx, prefix="node"),
                    role=role,
                    capacity=1.0 + (idx % 4) * 0.25,
                    inputs=[role.split("_")[0], "bus"],
                    outputs=[f"{role}_result", "health"],
                )
            )
        return {
            "node_count": len(nodes),
            "nodes": [node.to_record() for node in nodes],
            "routing": {
                "policy": "topic_and_capability_affinity",
                "processes_per_node": round(process_count / max(1, len(nodes)), 4),
                "fallback": "single_process_runtime",
            },
            "backends": capabilities.get("scale", {}),
        }


class EvaluationLab:
    """Frontier scorecards and stress-test plan."""

    benchmarks = [
        "concept_growth",
        "proof_chain_depth",
        "contradiction_repair",
        "goal_completion",
        "self_mod_shadow_gain",
        "memory_continuity",
        "tool_world_success",
        "social_calibration",
        "world_model_prediction",
        "distributed_throughput",
        "language_grounding",
        "emergence_pressure",
    ]

    def build(self, power_metrics: dict[str, Any], experiments: Iterable[dict[str, Any]]) -> dict[str, Any]:
        experiment_count = len(list(experiments))
        benchmark_records = []
        for idx, name in enumerate(self.benchmarks, start=1):
            benchmark_records.append(
                {
                    "benchmark_id": _stable_id(name, prefix="bench"),
                    "name": name,
                    "metric": self._metric(name),
                    "target": self._target(name, power_metrics, experiment_count),
                    "cadence": "every_release" if idx % 3 else "every_cycle_sample",
                }
            )
        return {
            "benchmark_count": len(benchmark_records),
            "benchmarks": benchmark_records,
            "scorecard": {
                "frontier_score": power_metrics.get("total_frontier_score", 0.0),
                "autonomy": power_metrics.get("autonomy_index", 0.0),
                "recursive_improvement": power_metrics.get("recursive_improvement_index", 0.0),
                "embodiment": power_metrics.get("embodiment_grounding_index", 0.0),
            },
            "red_team_protocols": [
                "counterfactual_goal_conflict",
                "self_mod_regression",
                "social_misread",
                "tool_sequence_failure",
                "memory_corruption",
            ],
        }

    def _metric(self, name: str) -> str:
        return {
            "concept_growth": "valid_new_relations_per_cycle",
            "proof_chain_depth": "max_supported_hops",
            "contradiction_repair": "time_to_refute_and_revise",
            "goal_completion": "durable_goals_completed",
            "self_mod_shadow_gain": "shadow_gain_minus_regression",
            "memory_continuity": "cross_session_recall_precision",
            "tool_world_success": "successful_tool_plans",
            "social_calibration": "operator_preference_fit",
            "world_model_prediction": "transition_prediction_accuracy",
            "distributed_throughput": "cognitive_events_per_second",
            "language_grounding": "reply_claims_with_support",
            "emergence_pressure": "multi_subsystem_novel_outputs",
        }[name]

    def _target(self, name: str, metrics: dict[str, Any], experiment_count: int) -> float:
        base = 0.5 + min(0.25, experiment_count / 100.0)
        if name == "proof_chain_depth":
            return float(metrics.get("reasoning_depth_budget", 32))
        if name == "concept_growth":
            return float(metrics.get("concept_capacity_projection", 1_000_000)) / 10_000.0
        return round(base, 4)


class RoadmapPlanner:
    """Concrete multi-phase roadmap for turning NG into a frontier lab."""

    phase_names = [
        "substrate_unification",
        "workspace_intensification",
        "autonomous_curriculum",
        "recursive_self_improvement",
        "embodied_tool_mastery",
        "social_collaboration",
        "distributed_scaling",
        "frontier_evaluation",
        "long_horizon_continuity",
        "open_world_research",
    ]

    def build(self, program_pressure: float) -> dict[str, Any]:
        phases = []
        for idx, name in enumerate(self.phase_names, start=1):
            phases.append(
                {
                    "phase": idx,
                    "name": name,
                    "objective": self._objective(name),
                    "exit_criteria": self._exit_criteria(name),
                    "pressure": round(_bounded(program_pressure + idx * 0.025), 4),
                }
            )
        return {
            "phase_count": len(phases),
            "phases": phases,
            "principle": "every phase must produce measurable capability, not vibes",
        }

    def _objective(self, name: str) -> str:
        return {
            "substrate_unification": "merge symbolic, neural, memory, and tool surfaces into one operating system",
            "workspace_intensification": "increase integrated information and useful broadcast diversity",
            "autonomous_curriculum": "turn intrinsic drives into self-generated training tasks",
            "recursive_self_improvement": "shadow-test and promote capability-improving mutations",
            "embodied_tool_mastery": "master digital-body affordances through feedback loops",
            "social_collaboration": "model collaborators and repair misunderstandings",
            "distributed_scaling": "parallelize cognition across agents and backends",
            "frontier_evaluation": "measure against hard capability and regression gates",
            "long_horizon_continuity": "maintain identity, goals, and memory across sessions",
            "open_world_research": "pursue self-directed discovery in unfamiliar domains",
        }[name]

    def _exit_criteria(self, name: str) -> list[str]:
        return [
            f"{name}_tests_pass",
            f"{name}_metric_improves",
            "full_regression_suite_passes",
        ]


class ResearchProgramEngine:
    """High-level Darwin NG research program.

    This is intentionally broad. It builds a frontier lab record from the live
    runtime rather than a static manifesto, so every cycle exposes the active
    operating system, memory fabric, simulations, experiments, embodiment,
    social models, distributed plan, evaluation suite, and roadmap.
    """

    def __init__(self) -> None:
        self.cos = CognitiveOperatingSystem()
        self.memory = MemoryFabric()
        self.world_sim = WorldSimulationLab()
        self.rsi = RecursiveImprovementLab()
        self.embodiment = EmbodimentLab()
        self.social = SocialIntelligenceLab()
        self.distributed = DistributedCognitionLab()
        self.evaluation = EvaluationLab()
        self.roadmap = RoadmapPlanner()

    def build(
        self,
        workspace: dict[str, Any],
        goals: list[Any],
        capabilities: dict[str, Any],
        knowledge: dict[str, Any],
        frontier_protocols: dict[str, Any],
        power_metrics: dict[str, Any],
        meta_learning: dict[str, Any],
    ) -> dict[str, Any]:
        cos = self.cos.build(workspace, goals, capabilities)
        memory = self.memory.build(knowledge, capabilities)
        world_lab = self.world_sim.build(frontier_protocols, capabilities)
        rsi = self.rsi.build(meta_learning, goals, power_metrics)
        embodiment = self.embodiment.build(frontier_protocols)
        social = self.social.build(frontier_protocols)
        distributed = self.distributed.build(capabilities, cos)
        evaluation = self.evaluation.build(power_metrics, rsi["experiments"])
        active_engines = self._active_engines(
            cos=cos,
            memory=memory,
            world_lab=world_lab,
            rsi=rsi,
            embodiment=embodiment,
            social=social,
            distributed=distributed,
            evaluation=evaluation,
        )
        emergence_index = self._emergence_index(
            cos=cos,
            memory=memory,
            world_lab=world_lab,
            rsi=rsi,
            embodiment=embodiment,
            social=social,
            distributed=distributed,
            evaluation=evaluation,
            power_metrics=power_metrics,
        )
        roadmap = self.roadmap.build(emergence_index)
        return {
            "scale": "frontier_lab",
            "created_at": time.time(),
            "active_engines": active_engines,
            "cognitive_operating_system": cos,
            "memory_fabric": memory,
            "world_simulation_lab": world_lab,
            "recursive_improvement_lab": rsi,
            "embodiment_lab": embodiment,
            "social_lab": social,
            "distributed_lab": distributed,
            "evaluation_lab": evaluation,
            "roadmap": roadmap,
            "emergence_index": emergence_index,
            "claim": "large-scale self-directed research architecture, not a chatbot wrapper",
        }

    def _active_engines(self, **labs: dict[str, Any]) -> list[dict[str, Any]]:
        engines = []
        for lab_name, lab in labs.items():
            engines.append(
                {
                    "engine_id": _stable_id(lab_name, prefix="engine"),
                    "name": lab_name,
                    "status": "active",
                    "surface_area": len(lab),
                    "heartbeat": "cycle",
                }
            )
        extra = [
            "attention_allocator",
            "drive_homeostat",
            "goal_lattice",
            "plan_weaver",
            "counterfactual_lab",
            "curriculum_generator",
            "mutation_shadow_runner",
            "capability_scorekeeper",
            "collaboration_protocol",
            "body_schema_mapper",
        ]
        for name in extra:
            engines.append(
                {
                    "engine_id": _stable_id(name, prefix="engine"),
                    "name": name,
                    "status": "active",
                    "surface_area": 3,
                    "heartbeat": "background",
                }
            )
        return engines

    def _emergence_index(
        self,
        *,
        cos: dict[str, Any],
        memory: dict[str, Any],
        world_lab: dict[str, Any],
        rsi: dict[str, Any],
        embodiment: dict[str, Any],
        social: dict[str, Any],
        distributed: dict[str, Any],
        evaluation: dict[str, Any],
        power_metrics: dict[str, Any],
    ) -> float:
        signals = [
            min(1.0, cos.get("process_count", 0) / 20.0),
            min(1.0, len(memory.get("tiers", {})) / 6.0),
            min(1.0, world_lab.get("simulation_count", 0) / 8.0),
            min(1.0, rsi.get("experiment_count", 0) / 12.0),
            min(1.0, embodiment.get("sensor_count", 0) / 10.0),
            min(1.0, social.get("agent_models", 0) / 6.0),
            min(1.0, distributed.get("node_count", 0) / 12.0),
            min(1.0, evaluation.get("benchmark_count", 0) / 12.0),
            float(power_metrics.get("total_frontier_score", 0.0) or 0.0),
        ]
        return round(sum(signals) / len(signals), 4)
