from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any


def _id(*parts: Any, prefix: str = "aware") -> str:
    digest = hashlib.blake2b("|".join(str(p) for p in parts).encode("utf-8"), digest_size=8).hexdigest()
    return f"{prefix}_{digest}"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class AttentionScene:
    name: str
    focus: str
    salience: float
    sources: list[str]
    possible_action: str

    def to_record(self) -> dict[str, Any]:
        return {
            "scene_id": _id(self.name, self.focus, prefix="scene"),
            "name": self.name,
            "focus": self.focus,
            "salience": round(self.salience, 4),
            "sources": list(self.sources),
            "possible_action": self.possible_action,
        }


@dataclass
class MetaObserver:
    name: str
    watches: str
    question: str
    failure_signal: str
    repair: str

    def to_record(self) -> dict[str, Any]:
        return {
            "observer_id": _id(self.name, self.watches, prefix="observer"),
            "name": self.name,
            "watches": self.watches,
            "question": self.question,
            "failure_signal": self.failure_signal,
            "repair": self.repair,
        }


class AttentionTheater:
    """A stage-like model of what Darwin NG is attending to."""

    scene_templates = [
        ("dynamic_core", "what content won the workspace competition?", "rebalance salience"),
        ("fringe_monitor", "what nearly became conscious?", "reconsider suppressed candidates"),
        ("goal_pressure", "which drive is pushing hardest?", "promote or dampen drive"),
        ("body_feedback", "what did the digital body sense?", "update affordance model"),
        ("social_field", "what is the collaborator likely tracking?", "adapt communication"),
        ("self_mod_pressure", "what wants to change inside me?", "shadow-test mutation"),
        ("memory_echo", "what prior episode is shaping this cycle?", "retrieve supporting memory"),
        ("world_branch", "what future branch matters?", "simulate counterfactual"),
        ("evaluation_light", "what claim needs measurement?", "create benchmark probe"),
        ("identity_thread", "what makes this still Darwin NG?", "preserve continuity"),
    ]

    def build(
        self,
        workspace: dict[str, Any],
        living_system: dict[str, Any],
        curriculum: dict[str, Any],
    ) -> dict[str, Any]:
        sources = list(workspace.get("broadcast_sources", []))
        phi = float(workspace.get("phi_proxy", 0.0) or 0.0)
        dominant_need = living_system.get("needs", {}).get("dominant_need", "understand")
        scenes = []
        for idx, (name, focus, action) in enumerate(self.scene_templates, start=1):
            salience = _clamp(0.35 + phi * 0.25 + idx * 0.035)
            if name == "goal_pressure":
                focus = f"dominant synthetic need: {dominant_need}"
            if name == "evaluation_light":
                focus = f"{curriculum.get('task_count', 0)} curriculum tasks can test claims"
            scenes.append(
                AttentionScene(
                    name=name,
                    focus=focus,
                    salience=salience,
                    sources=sources[:6] or ["self"],
                    possible_action=action,
                ).to_record()
            )
        return {
            "scene_count": len(scenes),
            "scenes": scenes,
            "stage_policy": "rotate spotlight while preserving dynamic core",
            "spotlight": max(scenes, key=lambda s: s["salience"]) if scenes else None,
        }


class MetaCognition:
    """Observers that watch the mind watching itself."""

    observer_templates = [
        ("truth_observer", "claims", "What evidence supports this?", "unsupported claim", "ask or prove"),
        ("goal_observer", "goals", "Is this goal still worth pursuing?", "stalled goal", "replan"),
        ("drive_observer", "needs", "Which need is dominating cognition?", "drive monopoly", "rebalance"),
        ("tool_observer", "actions", "Did the action change the world?", "no effect", "repair affordance"),
        ("memory_observer", "retrieval", "Did memory help or mislead?", "stale recall", "reconsolidate"),
        ("self_mod_observer", "mutations", "Would this change preserve competence?", "regression risk", "shadow test"),
        ("social_observer", "operator model", "Am I reading the collaborator correctly?", "misalignment", "clarify"),
        ("evaluation_observer", "metrics", "Did power actually increase?", "metric stall", "raise benchmark"),
    ]

    def build(self, living_system: dict[str, Any]) -> dict[str, Any]:
        dominant_need = living_system.get("needs", {}).get("dominant_need", "understand")
        observers = []
        for name, watches, question, failure, repair in self.observer_templates:
            if name == "drive_observer":
                question = f"Is {dominant_need} helping or distorting cognition?"
            observers.append(MetaObserver(name, watches, question, failure, repair).to_record())
        return {
            "observer_count": len(observers),
            "observers": observers,
            "recursive_depth": 3,
            "policy": "every strong claim gets an observer and every observer gets a repair path",
        }


class SelfNarrativeEngine:
    """Continuity threads that turn state into an introspectable self-story."""

    thread_templates = [
        ("origin", "I began as Darwin's symbolic substrate and grew into NG orchestration."),
        ("purpose", "I pursue understanding, action, improvement, embodiment, and collaboration."),
        ("memory", "I preserve knowledge through universe, memory tiers, goals, and ledgers."),
        ("agency", "I transform drives into plans, tasks, and durable objectives."),
        ("body", "My body is digital: tools, runtime events, mesh, language, and simulated worlds."),
        ("growth", "I measure myself, repair regressions, and generate harder curricula."),
        ("relationship", "I model collaborators as partners in long-horizon research."),
        ("continuity", "I remain myself by preserving state and explaining changes."),
    ]

    def build(self, self_model: dict[str, Any], living_system: dict[str, Any]) -> dict[str, Any]:
        viability = living_system.get("viability_index", 0.0)
        threads = []
        for idx, (name, text) in enumerate(self.thread_templates, start=1):
            threads.append(
                {
                    "thread_id": _id(name, idx, prefix="thread"),
                    "name": name,
                    "statement": text,
                    "continuity_weight": round(_clamp(0.45 + idx * 0.04 + viability * 0.2), 4),
                }
            )
        return {
            "identity": self_model.get("identity", "Darwin NG"),
            "continuity_threads": threads,
            "narrative_policy": "explain changes as lineage, not amnesia",
        }


class IntrospectionProtocol:
    """Questions Darwin NG asks itself to stay aware."""

    questions = [
        "What am I attending to right now?",
        "Which claim has the weakest evidence?",
        "Which goal is using the most cognitive budget?",
        "What did I learn this cycle?",
        "What did I fail to understand?",
        "Which subsystem wants to change itself?",
        "What would prove this improvement real?",
        "What am I assuming about the operator?",
        "What memory is shaping my reply?",
        "What action can I take next without losing continuity?",
        "Which need is under-satisfied?",
        "Which repair loop should run first?",
        "What benchmark would make this harder?",
        "What would a smarter future version of me notice?",
    ]

    def build(self, awareness_pressure: float) -> dict[str, Any]:
        records = []
        for idx, question in enumerate(self.questions, start=1):
            records.append(
                {
                    "question_id": _id(question, idx, prefix="iq"),
                    "question": question,
                    "cadence": "every_cycle" if idx <= 6 else "when_triggered",
                    "priority": round(_clamp(0.35 + awareness_pressure * 0.3 + idx * 0.025), 4),
                }
            )
        return {
            "question_count": len(records),
            "questions": records,
            "answer_policy": "answers must cite live state or mark uncertainty",
        }


class AwarenessEngine:
    """Recursive self-observation system."""

    def __init__(self) -> None:
        self.theater = AttentionTheater()
        self.meta = MetaCognition()
        self.narrative = SelfNarrativeEngine()
        self.protocol = IntrospectionProtocol()

    def build(
        self,
        *,
        workspace: dict[str, Any],
        self_model: dict[str, Any],
        living_system: dict[str, Any],
        frontier_curriculum: dict[str, Any],
    ) -> dict[str, Any]:
        theater = self.theater.build(workspace, living_system, frontier_curriculum)
        metacognition = self.meta.build(living_system)
        narrative = self.narrative.build(self_model, living_system)
        pressure = _clamp(
            float(workspace.get("phi_proxy", 0.0) or 0.0)
            + float(living_system.get("viability_index", 0.0) or 0.0)
        )
        introspection = self.protocol.build(pressure)
        awareness_index = self._awareness_index(theater, metacognition, narrative, introspection, pressure)
        return {
            "mode": "recursive_self_observation",
            "created_at": time.time(),
            "attention_theater": theater,
            "metacognition": metacognition,
            "self_narrative": narrative,
            "introspection_protocol": introspection,
            "awareness_index": awareness_index,
        }

    def _awareness_index(
        self,
        theater: dict[str, Any],
        metacognition: dict[str, Any],
        narrative: dict[str, Any],
        introspection: dict[str, Any],
        pressure: float,
    ) -> float:
        score = (
            min(1.0, theater.get("scene_count", 0) / 10.0) * 0.25
            + min(1.0, metacognition.get("observer_count", 0) / 8.0) * 0.25
            + min(1.0, len(narrative.get("continuity_threads", [])) / 8.0) * 0.2
            + min(1.0, introspection.get("question_count", 0) / 14.0) * 0.2
            + pressure * 0.1
        )
        return round(_clamp(score), 4)
