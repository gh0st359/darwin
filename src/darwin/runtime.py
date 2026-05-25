from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from darwin.agent import Darwin
from darwin.composer import NaturalLanguageComposer
from darwin.critic import Critique, ResponseCritic
from darwin.discourse import DiscoursePlanner, ResponsePlan
from darwin.dlm import DLMRenderResult, DarwinLanguageModule, StubDLM
from darwin.embodiment import ConversationAdapter, EnvironmentAdapter
from darwin.experiments import ExperimentResult
from darwin.instrumentation import BackgroundLogEntry, PlanLogEntry, StructuredLogger
from darwin.retrieval import ContextRetriever, RetrievalPacket
from darwin.self_modification import ModificationOutcome, SelfModificationEngine
from darwin.storage import PersistentStore
from darwin.thought import ThoughtTrace
from darwin.training_data import TrainingDataCollector
from darwin.types import Action, Goal, Transition


@dataclass
class RuntimeEvent:
    kind: str
    content: str
    payload: dict[str, Any] = field(default_factory=dict)
    loop: str = "main"
    timestamp: float = field(default_factory=time.time)


@dataclass
class BackgroundLoopSpec:
    name: str
    interval: float
    runner: Callable[[], RuntimeEvent | None]


class DarwinRuntime:
    """Always-on, multi-threaded local cognition loop for Darwin.

    Phase 2 redesign: each cognitive activity runs in its own background
    thread on its own cadence. Threads share Darwin via a single lock,
    and every wake-up emits a visible RuntimeEvent + a structured log
    entry so the mind is observable even when no one is talking to it.
    """

    def __init__(
        self,
        darwin: Darwin,
        adapter: EnvironmentAdapter,
        goal: Goal,
        store: PersistentStore | None = None,
        interval: float = 2.0,
        event_sink: Callable[[RuntimeEvent], None] | None = None,
        logger: StructuredLogger | None = None,
        dlm: DarwinLanguageModule | None = None,
        training_collector: TrainingDataCollector | None = None,
        state_path: str | Path | None = "darwin_runtime_state.json",
        loop_intervals: dict[str, float] | None = None,
    ) -> None:
        self.darwin = darwin
        self.adapter = adapter
        self.goal = goal
        self.store = store or darwin.store
        self.interval = interval
        self.conversation = ConversationAdapter()
        self.retriever = ContextRetriever()
        self.discourse = DiscoursePlanner()
        self.composer = NaturalLanguageComposer()
        self.critic = ResponseCritic()
        self.events: list[RuntimeEvent] = []
        self.event_sink = event_sink
        self.stream_enabled = True
        self.logger = logger or StructuredLogger()
        self.dlm: DarwinLanguageModule = dlm or StubDLM()
        self.training_collector = training_collector or TrainingDataCollector()
        self.self_mod_engine = SelfModificationEngine(darwin)
        self.last_thought_trace: ThoughtTrace | None = None
        self.last_retrieval: RetrievalPacket | None = None
        self.last_response_plan: ResponsePlan | None = None
        self.last_critique: Critique | None = None
        self.last_render: DLMRenderResult | None = None
        self.last_self_mod_outcomes: list[ModificationOutcome] = []
        self.last_simulation: dict[str, Any] | None = None
        self.last_consolidation: dict[str, Any] | None = None
        self.last_uncertainty_scan: dict[str, Any] | None = None
        self._attention_queue: deque[dict[str, Any]] = deque(maxlen=50)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._threads: dict[str, threading.Thread] = {}
        self._loop_state: dict[str, dict[str, Any]] = {}
        self.state_path = Path(state_path) if state_path else None
        defaults = {
            "experiment": interval,
            "simulation": max(2.5, interval * 1.5),
            "dream": max(8.0, interval * 4.0),
            "self_modification": max(15.0, interval * 6.0),
            "uncertainty": max(6.0, interval * 3.0),
        }
        if loop_intervals:
            defaults.update(loop_intervals)
        self.loop_intervals = defaults
        self._load_state()

    @property
    def running(self) -> bool:
        return any(thread.is_alive() for thread in self._threads.values())

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        specs = [
            BackgroundLoopSpec("experiment", self.loop_intervals["experiment"], self._loop_experiment),
            BackgroundLoopSpec("simulation", self.loop_intervals["simulation"], self._loop_simulation),
            BackgroundLoopSpec("dream", self.loop_intervals["dream"], self._loop_dream),
            BackgroundLoopSpec("self_modification", self.loop_intervals["self_modification"], self._loop_self_modification),
            BackgroundLoopSpec("uncertainty", self.loop_intervals["uncertainty"], self._loop_uncertainty),
        ]
        for spec in specs:
            thread = threading.Thread(
                target=self._driver,
                args=(spec,),
                name=f"darwin-{spec.name}",
                daemon=True,
            )
            self._threads[spec.name] = thread
            thread.start()
        self._event(
            "runtime",
            f"Darwin's continuous cognition is running across {len(self._threads)} background loops.",
            payload={"loops": list(self._threads)},
            loop="main",
        )

    def stop(self) -> None:
        self._stop.set()
        for thread in list(self._threads.values()):
            thread.join(timeout=max(1.0, self.interval * 2.0))
        self._threads.clear()
        self._save_state()
        self._event("runtime", "Darwin's continuous cognition loops stopped.", loop="main")

    def _driver(self, spec: BackgroundLoopSpec) -> None:
        # Stagger startup to avoid thundering herd
        delay = self._loop_state.get(spec.name, {}).get("phase_offset", 0.0)
        if delay:
            self._stop.wait(min(delay, spec.interval))
        while not self._stop.wait(spec.interval):
            start = time.perf_counter()
            try:
                event = spec.runner()
            except Exception as exc:  # pragma: no cover - defensive
                event = self._event(
                    "error",
                    f"loop {spec.name} failed: {exc!r}",
                    payload={"loop": spec.name},
                    loop=spec.name,
                )
            duration_ms = (time.perf_counter() - start) * 1000.0
            if event is not None:
                self.logger.log_background(
                    BackgroundLogEntry(
                        loop=spec.name,
                        kind=event.kind,
                        content=event.content,
                        payload=event.payload,
                        duration_ms=duration_ms,
                    )
                )
                self._loop_state.setdefault(spec.name, {})["last_event"] = event.kind
                self._loop_state[spec.name]["last_time"] = time.time()
        # Final state checkpoint per loop
        self._loop_state.setdefault(spec.name, {})["stopped"] = True

    # -- background loops ------------------------------------------------

    def _loop_experiment(self) -> RuntimeEvent | None:
        with self._lock:
            state = self.adapter.observe()
            attention = self._next_attention()
            actions = self._actions_for_attention(attention) if attention is not None else []
            if not actions:
                actions = self.adapter.possible_actions()
            variable_filter = self._variable_filter_for_actions(actions)
            proposals = self.darwin.experiment_engine.propose(
                state,
                actions,
                goal=self.goal,
                limit=max(3, len(actions)),
                variable_filter=variable_filter,
                variable_filter_for_action=self._variable_filter_for_action,
            )
            proposal = self._select_experiment_proposal(proposals, attention)

            if proposal is not None:
                mode = (
                    "active_experiment"
                    if proposal.uncertainty >= 0.25
                    else "attention_experiment"
                    if attention is not None
                    else "maintenance_experiment"
                )
                before = self.adapter.observe()
                after, reward = self.adapter.apply(proposal.action)
                transition = Transition(
                    before=before,
                    action=proposal.action.name,
                    after=after,
                    reward=reward,
                    t=self._next_time(),
                    metadata=self._metadata_for_action(proposal.action, {
                        "loop": "experiment",
                        "question": proposal.question,
                        "predicted_state": proposal.predicted_state,
                        "expected_reward": proposal.expected_reward,
                        "mode": mode,
                        **(attention or {}),
                    }),
                )
                self.darwin.learn(transition)
                result = self.darwin.experiment_engine.evaluate(proposal, transition)
                if self.store is not None:
                    self.store.record_experiment(result.to_record())
                return self._experiment_event(result, loop="experiment")

            reflection = self.darwin.reflect()
            return self._event("reflection", reflection, loop="experiment")

    def _loop_simulation(self) -> RuntimeEvent | None:
        with self._lock:
            state = self.adapter.observe()
            actions = self.adapter.possible_actions()
            chains = self.darwin.planner.reason_chain(state, actions, depth=3, beam=4)
            if not chains:
                return None
            best = chains[0]
            # Highest-uncertainty single step inside the simulated chain becomes
            # a learning signal: it's escalated to the experiment loop next time
            # by registering it in the prediction-failure counter the SelfModel
            # uses to drive learning_priority.
            worst_node = max(best.nodes, key=lambda node: node.uncertainty, default=None)
            if worst_node is not None and worst_node.uncertainty >= 0.5:
                key = f"{worst_node.action}:simulation_uncertainty"
                self.darwin.self_model.prediction_failures[key] += 1
            self.last_simulation = best.to_record()
            content = (
                f"Mental simulation: {best.describe()} "
                f"(chain conf {best.chain_confidence:.2f}, "
                f"reward {best.total_expected_reward:.2f})"
            )
            return self._event(
                "simulation",
                content,
                payload={"chain": best.to_record()},
                loop="simulation",
            )

    def _loop_dream(self) -> RuntimeEvent | None:
        with self._lock:
            reflection = self.darwin.reflect()
            consolidation = self.darwin.memory.concepts.consolidate()
            concepts = self.darwin.memory.concepts.salient(limit=5)
            concept_records = [concept.to_record() for concept in concepts]
            episodes_count = len(self.darwin.memory.episodes)
            top_unknown = list(self.darwin.semantic_memory.unknown_terms.most_common(5))
            self.last_consolidation = {
                "reflection": reflection,
                "concepts": concept_records,
                "episodes": episodes_count,
                "unknown_terms": top_unknown,
                "clusters_formed": consolidation.get("clusters_formed", []),
                "concepts_decayed": consolidation.get("concepts_decayed", []),
            }
            concept_line = ", ".join(concept.name for concept in concepts) or "no concepts yet"
            cluster_note = (
                f" Formed {len(consolidation['clusters_formed'])} concept clusters."
                if consolidation.get("clusters_formed")
                else ""
            )
            content = f"Dreaming. {reflection} Salient concepts: {concept_line}.{cluster_note}"
            return self._event(
                "dream",
                content,
                payload={"consolidation": self.last_consolidation},
                loop="dream",
            )

    def _loop_self_modification(self) -> RuntimeEvent | None:
        with self._lock:
            outcomes = self.self_mod_engine.run_cycle()
            self.last_self_mod_outcomes = outcomes
            accepted = [outcome for outcome in outcomes if outcome.accepted]
            rejected = [outcome for outcome in outcomes if not outcome.accepted]
            if accepted:
                desc = ", ".join(
                    f"{outcome.proposal.kind} (gain {outcome.improvement:.3f})"
                    for outcome in accepted
                )
                content = f"Self-modification adopted: {desc}."
            elif rejected:
                desc = ", ".join(outcome.proposal.kind for outcome in rejected[:3])
                content = f"Self-modification proposed but rejected: {desc}."
            else:
                content = "Self-modification: no useful proposals this cycle."
            for outcome in outcomes:
                if self.store is not None:
                    self.store.record_self_modification(outcome.to_record())
            return self._event(
                "self_modification",
                content,
                payload={"outcomes": [outcome.to_record() for outcome in outcomes]},
                loop="self_modification",
            )

    def _loop_uncertainty(self) -> RuntimeEvent | None:
        with self._lock:
            state = self.adapter.observe()
            actions = self.adapter.possible_actions()
            scan = []
            for action in actions:
                uncertainty = self.darwin.causal_model.uncertainty_for(state, action.name)
                scan.append({"action": action.name, "uncertainty": uncertainty})
            scan.sort(key=lambda item: item["uncertainty"], reverse=True)
            self.last_uncertainty_scan = {"scan": scan, "state": state}
            top = scan[:3]
            content = "Uncertainty scan: " + "; ".join(
                f"{item['action']}={item['uncertainty']:.2f}" for item in top
            )
            return self._event(
                "uncertainty",
                content,
                payload={"scan": scan},
                loop="uncertainty",
            )

    # -- on-demand cognition --------------------------------------------

    def cognition_cycle(self) -> RuntimeEvent:
        event = self._loop_experiment()
        return event if event is not None else self._event("reflection", self.darwin.reflect(), loop="manual")

    def dream(self) -> RuntimeEvent:
        event = self._loop_dream()
        return event if event is not None else self._event("dream", "no consolidation produced", loop="manual")

    def run_self_modification(self) -> list[ModificationOutcome]:
        self._loop_self_modification()
        return self.last_self_mod_outcomes

    def run_simulation(self) -> dict[str, Any] | None:
        self._loop_simulation()
        return self.last_simulation

    # -- conversation ----------------------------------------------------

    def chat(self, message: str) -> str:
        with self._lock:
            if self.store is not None:
                self.store.record_chat("user", message)

            user_frame = self.darwin.interpret_language(message, source="user")
            response = self._respond(message, user_frame)
            darwin_frame = self.darwin.interpret_language(response, source="darwin")
            transition = self.conversation.make_transition(message, response, t=self._next_time())
            self._remember_attention_from_frame(user_frame)
            transition = Transition(
                before=transition.before,
                action=transition.action,
                after=transition.after,
                reward=transition.reward,
                t=transition.t,
                metadata={
                    **dict(transition.metadata),
                    "user_semantics": user_frame.to_record(),
                    "darwin_semantics": darwin_frame.to_record(),
                },
            )
            self.darwin.learn(transition)

            if self.store is not None:
                self.store.record_chat("darwin", response)

            self._event(
                "chat",
                response,
                payload={"message_signal": transition.before, "response_signal": transition.after},
                loop="main",
            )
            return response

    def recent_events(self, limit: int = 20) -> list[RuntimeEvent]:
        return self.events[-limit:]

    def set_streaming(self, enabled: bool) -> None:
        self.stream_enabled = enabled

    def use_dlm(self, dlm: DarwinLanguageModule) -> None:
        self.dlm = dlm

    # -- internal --------------------------------------------------------

    def _respond(self, message: str, semantic_frame) -> str:
        trace = ThoughtTrace(user_text=message, semantic_summary=semantic_frame.summary())
        trace.add(
            "parse",
            semantic_frame.summary(),
            confidence=semantic_frame.confidence,
            evidence=[semantic_frame.original_text],
        )

        retrieval = self.retriever.retrieve(
            self.darwin,
            semantic_frame,
            recent_events=self.recent_events(limit=8),
        )
        trace.add(
            "retrieve",
            retrieval.summary(),
            confidence=0.5 if retrieval.items else 0.2,
            evidence=[item.content for item in retrieval.top(3)],
        )

        plan = self.discourse.plan(
            frame=semantic_frame,
            packet=retrieval,
            darwin=self.darwin,
            adapter=self.adapter,
            goal=self.goal,
            recent_events=self.recent_events(limit=5),
        )
        trace.add(
            "plan",
            f"{plan.mode}: {plan.intent}",
            confidence=plan.confidence,
            evidence=plan.answer_points[:3],
            payload={"plan_id": plan.plan_id, "tone": plan.tone},
        )

        render = self.dlm.render(plan, semantic_frame, trace)
        draft = render.text
        self.last_render = render
        trace.add(
            "dlm",
            f"renderer={render.renderer} valid={render.valid}",
            confidence=0.6 if render.valid else 0.3,
            evidence=render.validation_notes,
        )
        if not render.valid:
            # Faithfulness failure: fall back to the deterministic composer.
            draft = self.composer.compose(plan, semantic_frame, trace)
            trace.add(
                "dlm_fallback",
                "DLM output rejected; falling back to deterministic composer.",
                confidence=0.5,
                evidence=render.validation_notes,
            )

        critique = self.critic.evaluate(plan, draft, semantic_frame, retrieval)
        if not critique.passed:
            trace.add(
                "critic",
                critique.summary(),
                confidence=0.45,
                evidence=critique.revisions,
            )
            plan = self.critic.revise(plan, critique, semantic_frame, retrieval)
            render = self.dlm.render(plan, semantic_frame, trace)
            draft = render.text if render.valid else self.composer.compose(plan, semantic_frame, trace)
            self.last_render = render
            critique = self.critic.evaluate(plan, draft, semantic_frame, retrieval)
        else:
            trace.add("critic", "response passed self-critique", confidence=0.75)

        trace.final_mode = plan.mode
        trace.final_confidence = plan.confidence
        self.last_thought_trace = trace
        self.last_retrieval = retrieval
        self.last_response_plan = plan
        self.last_critique = critique

        if self.store is not None:
            self.store.record_thought("thought_trace", trace.compact(), trace.to_record())
            self.store.record_thought("response_plan", plan.thesis, plan.to_record())
            self.store.record_thought("response_critique", critique.summary(), critique.to_record())

        self.logger.log_plan(
            PlanLogEntry(
                plan_id=plan.plan_id,
                user_text=message,
                semantic_summary=semantic_frame.summary(),
                plan=plan.to_record(),
                rendering=draft,
                critique=critique.to_record(),
                trace=trace.to_record(),
                renderer=render.renderer if render else "composer",
                background=False,
            )
        )
        self.training_collector.add(
            plan_id=plan.plan_id,
            user_text=message,
            plan_payload=plan.to_dlm_payload(),
            rendering=draft,
            renderer=render.renderer if render else "composer",
            critique_passed=critique.passed,
        )

        self._event(
            "thought",
            trace.compact(),
            payload={
                "trace": trace.to_record(),
                "retrieval": retrieval.to_record(),
                "plan": plan.to_record(),
                "critique": critique.to_record(),
                "dlm_payload": plan.to_dlm_payload(),
                "renderer": render.renderer if render else "composer",
            },
            loop="main",
        )
        return draft

    def _experiment_event(self, result: ExperimentResult, loop: str = "main") -> RuntimeEvent:
        domain = result.transition.metadata.get("domain")
        domain_prefix = f"[{domain}] " if domain else ""
        if result.confirmed:
            content = f"{domain_prefix}Experiment confirmed: {result.proposal.question}"
        else:
            surprise_list = ", ".join(result.surprises)
            content = f"{domain_prefix}Experiment produced surprise in {surprise_list}: {result.proposal.question}"
        return self._event("experiment", content, payload=result.to_record(), loop=loop)

    def _metadata_for_action(self, action: Action, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata: dict[str, Any] = {"scope": "world", "world": getattr(self.adapter, "name", "unknown")}
        action_metadata = getattr(self.adapter, "action_metadata", None)
        if callable(action_metadata):
            metadata.update(dict(action_metadata(action)))
        metadata.update(dict(action.metadata))
        if "vocabulary" in metadata:
            metadata["vocabulary"] = sorted(metadata["vocabulary"])
        if extra:
            metadata.update(extra)
        metadata.setdefault("domain", metadata.get("world", "world"))
        return metadata

    def _select_experiment_proposal(
        self,
        proposals: list[Any],
        attention: dict[str, Any] | None,
    ) -> Any | None:
        if not proposals:
            return None
        if proposals[0].uncertainty >= 0.25 or attention is not None:
            return proposals[0]
        return sorted(
            proposals,
            key=lambda proposal: (
                self.darwin.causal_model.action_count(proposal.action.name),
                str(proposal.action.metadata.get("domain", "")),
                proposal.action.name,
            ),
        )[0]

    def _variable_filter_for_actions(self, actions: list[Action]) -> Callable[[str], bool] | None:
        domains = {
            str(action.metadata.get("domain", ""))
            for action in actions
            if action.metadata.get("domain")
        }
        if len(domains) != 1:
            return None
        domain = next(iter(domains))
        variable_finder = getattr(self.adapter, "variables_for_domain", None)
        if not callable(variable_finder):
            return None
        variables = set(variable_finder(domain))
        if not variables:
            return None
        return variables.__contains__

    def _variable_filter_for_action(self, action: Action, variable: str) -> bool:
        domain = str(action.metadata.get("domain", ""))
        if not domain:
            return True
        variable_finder = getattr(self.adapter, "variables_for_domain", None)
        if not callable(variable_finder):
            return True
        variables = set(variable_finder(domain))
        return variable in variables if variables else True

    def _remember_attention_from_frame(self, frame) -> None:
        terms = set(frame.tokens)
        for grounding in frame.groundings:
            terms.add(grounding.name)
            terms.update(grounding.name.replace("_", " ").replace("/", " ").replace(".", " ").split())
        for proposition in frame.hypotheses or frame.propositions:
            terms.update(proposition.subject.split())
            terms.update(proposition.object.split())
            terms.add(proposition.relation)
        if frame.speech_act in {"teaching", "hypothesis", "goal", "correction", "directive"} and terms:
            self._attention_queue.append(
                {
                    "attention_source": (
                        "semantic_hypothesis"
                        if frame.speech_act in {"teaching", "hypothesis"} or frame.hypotheses or frame.propositions
                        else "semantic_focus"
                    ),
                    "attention_terms": sorted(terms)[:40],
                    "attention_text": frame.original_text,
                }
            )

    def _next_attention(self) -> dict[str, Any] | None:
        while self._attention_queue:
            attention = self._attention_queue.popleft()
            terms = set(attention.get("attention_terms", []))
            if self._actions_for_attention(attention):
                return {**attention, "attention_terms": sorted(terms)}
        return None

    def _actions_for_attention(self, attention: dict[str, Any] | None) -> list[Action]:
        if attention is None:
            return []
        terms = set(str(term).lower() for term in attention.get("attention_terms", []))
        finder = getattr(self.adapter, "actions_for_terms", None)
        if callable(finder):
            matches = finder(terms)
            if matches:
                return matches
        return []

    def _event(
        self,
        kind: str,
        content: str,
        payload: dict[str, Any] | None = None,
        loop: str = "main",
    ) -> RuntimeEvent:
        event = RuntimeEvent(kind=kind, content=content, payload=payload or {}, loop=loop)
        self.events.append(event)
        if len(self.events) > 500:
            self.events = self.events[-500:]
        if self.store is not None and kind != "chat":
            self.store.record_thought(kind, content, payload or {})
        if self.event_sink is not None and self.stream_enabled:
            self.event_sink(event)
        return event

    def _next_time(self) -> int:
        value = getattr(self.darwin, "_time", 0)
        setattr(self.darwin, "_time", value + 1)
        return value

    # -- persistent runtime state ---------------------------------------

    def _load_state(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            with self.state_path.open("r", encoding="utf-8") as handle:
                snapshot = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        self._loop_state = snapshot.get("loops", {})
        time_value = snapshot.get("darwin_time")
        if isinstance(time_value, int):
            setattr(self.darwin, "_time", time_value)
        exploration = snapshot.get("exploration_rate")
        if isinstance(exploration, (int, float)):
            self.darwin.exploration_rate = float(exploration)
        min_samples = snapshot.get("min_samples")
        if isinstance(min_samples, int):
            self.darwin.causal_model.min_samples = min_samples
        planner = snapshot.get("planner_overrides")
        if isinstance(planner, dict):
            setattr(self.darwin, "_planner_overrides", dict(planner))

    def _save_state(self) -> None:
        if not self.state_path:
            return
        snapshot = {
            "loops": self._loop_state,
            "darwin_time": getattr(self.darwin, "_time", 0),
            "exploration_rate": self.darwin.exploration_rate,
            "min_samples": self.darwin.causal_model.min_samples,
            "planner_overrides": dict(getattr(self.darwin, "_planner_overrides", {})),
            "saved_at": time.time(),
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with self.state_path.open("w", encoding="utf-8") as handle:
                json.dump(snapshot, handle, indent=2)
        except OSError:
            pass


def ensure_chat_action(actions: list[Action]) -> list[Action]:
    if any(action.name == "chat_with_user" for action in actions):
        return actions
    return [
        *actions,
        Action("chat_with_user", cost=0.0, description="Exchange language with the user as experience."),
    ]
