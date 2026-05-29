"""ConceptualWorld — Darwin's universe presented as a World it can act in.

The v5 brain ran inside a toy six-variable room (curtains, switch, fuse, …).
That world was fine for testing the substrate but produced a brain that
literally only knew about curtains. The default world is now ``Conceptual
World``: an adapter exposing Darwin's own ``ConceptUniverse`` through the
existing ``World`` protocol (``observe`` / ``possible_actions`` / ``apply``)
so every cognitive loop — experiment, simulation, dream, self-modification,
interior simulation — runs *inside concept space*.

Actions available to the brain:

  * ``explore(concept)`` — visit a concept and surface its neighborhood.
  * ``compose(a, b)`` — propose a derived parent kind for two concepts.
  * ``generalize(concept)`` — abstract upward from a concept.
  * ``specialize(concept)`` — find or propose a specific instance.
  * ``analogize(concept)`` — hunt for a cross-domain mirror.
  * ``reflect(concept)`` — emit a first-person reflection.
  * ``derive`` — run a pass of the ConceptDeriver against current state.

Every action returns a ``Transition`` whose ``after`` state encodes the
post-action snapshot (concept currently in focus, neighbor count, domain
mix, derivation count). Rewards are small *epistemic* signals: any action
that *reduces local uncertainty* (links a previously isolated concept,
adds a new derived concept, finds a fresh bridge) gets a positive reward;
no-ops get zero; failed actions get a slight negative reward.

This world has no "curtains". It has whatever Darwin has been thinking
about. As the universe grows from chat and derivation, so does the world
Darwin can act in.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from darwin.types import Action, State, Transition
from darwin.universe.concept_universe import ConceptUniverse
from darwin.universe.derivation import ConceptDeriver


CONCEPTUAL_ACTIONS: list[Action] = [
    Action("explore_concept", cost=0.0, description="visit a concept and surface its neighborhood"),
    Action("compose_concepts", cost=0.0, description="combine two related concepts into a candidate parent kind"),
    Action("generalize_concept", cost=0.0, description="propose a more general kind for the focused concept"),
    Action("specialize_concept", cost=0.0, description="find or propose a specific instance of the focused kind"),
    Action("analogize_concept", cost=0.0, description="seek a cross-domain analogue of the focused concept"),
    Action("reflect_concept", cost=0.0, description="generate a first-person reflection on the focused concept"),
    Action("derive_concepts", cost=0.0, description="run a derivation pass over current regularities"),
    Action("wander_universe", cost=0.0, description="random walk to find an under-visited region"),
]


@dataclass
class _Focus:
    primary: str = "self"
    secondary: str = ""
    last_action: str = ""
    last_reward: float = 0.0
    last_summary: str = ""
    last_changed_at: float = field(default_factory=time.time)


class ConceptualWorld:
    """Adapter exposing Darwin's concept universe as a World."""

    def __init__(
        self,
        universe: ConceptUniverse,
        *,
        deriver: ConceptDeriver | None = None,
        seed: int = 73,
    ) -> None:
        self.universe = universe
        self.deriver = deriver
        self._rng = random.Random(seed)
        self._focus = _Focus(primary=self._pick_initial_focus())
        self._step = 0
        self._action_count = 0
        self._derivations_seen = 0

    # -- World protocol ----------------------------------------------------

    def observe(self) -> State:
        focus = self._focus
        concept = self.universe.get(focus.primary)
        if concept is None:
            # The universe may have just been wiped; refocus on the most
            # salient concept available.
            focus.primary = self._pick_initial_focus()
            concept = self.universe.get(focus.primary)
        neighbors = self.universe.neighbors(focus.primary) if concept else []
        domains = sorted({
            self.universe.expect(rel.target).domain for rel in neighbors
            if self.universe.has(rel.target)
        })
        summary = self.universe.summary()
        return {
            "focus": focus.primary,
            "secondary_focus": focus.secondary,
            "focus_domain": concept.domain if concept else "",
            "neighbors": len(neighbors),
            "neighbor_domains": len(domains),
            "concept_count": int(summary["concepts"]),
            "relation_count": int(summary["relations"]),
            "domain_count": int(summary["domains"]),
            "last_action": focus.last_action,
            "last_reward": round(focus.last_reward, 3),
            "step": self._step,
        }

    def possible_actions(self) -> list[Action]:
        return list(CONCEPTUAL_ACTIONS)

    def apply(self, action: Action) -> tuple[State, float]:
        self._action_count += 1
        self._step += 1
        name = getattr(action, "name", str(action))
        before_relations = self.universe.summary()["relations"]
        before_concepts = self.universe.summary()["concepts"]
        reward = 0.0
        summary = ""
        if name == "explore_concept":
            summary, reward = self._explore()
        elif name == "compose_concepts":
            summary, reward = self._compose()
        elif name == "generalize_concept":
            summary, reward = self._generalize()
        elif name == "specialize_concept":
            summary, reward = self._specialize()
        elif name == "analogize_concept":
            summary, reward = self._analogize()
        elif name == "reflect_concept":
            summary, reward = self._reflect()
        elif name == "derive_concepts":
            summary, reward = self._derive()
        elif name == "wander_universe":
            summary, reward = self._wander()
        else:
            summary, reward = f"unknown action {name!r}", -0.1

        # Epistemic reward bonus: did this action grow the graph?
        after_relations = self.universe.summary()["relations"]
        after_concepts = self.universe.summary()["concepts"]
        growth_reward = (
            0.2 * (after_concepts - before_concepts)
            + 0.05 * (after_relations - before_relations)
        )
        reward += growth_reward
        self._focus.last_action = name
        self._focus.last_reward = reward
        self._focus.last_summary = summary
        return self.observe(), reward

    def make_transition(self, before: State, after: State, *, reward: float) -> Transition:
        return Transition(
            before=before,
            action=self._focus.last_action,
            after=after,
            reward=reward,
            t=self._step,
            metadata={
                "track": "grounded",
                "world": "conceptual",
                "summary": self._focus.last_summary,
            },
        )

    # -- action implementations ------------------------------------------

    def _pick_initial_focus(self) -> str:
        if self.universe.has("self"):
            return "self"
        all_concepts = self.universe.all_concepts()
        if not all_concepts:
            return ""
        return all_concepts[0].name

    def _pick_next_focus(self) -> str:
        """Pick a concept to focus on next — prefer recently grown or salient."""

        focus = self._focus.primary
        neighbors = self.universe.neighbors(focus) if focus else []
        if neighbors:
            chosen = self._rng.choices(
                neighbors,
                weights=[max(0.01, rel.weight) for rel in neighbors],
                k=1,
            )[0]
            return chosen.target
        all_concepts = self.universe.all_concepts()
        if not all_concepts:
            return focus
        return self._rng.choice(all_concepts).name

    def _explore(self) -> tuple[str, float]:
        focus = self._focus.primary
        if not focus or not self.universe.has(focus):
            return "no focus to explore", -0.05
        nbhd = self.universe.neighborhood(focus, hops=1, max_nodes=8)
        # Move focus to a neighbor on every explore so the brain doesn't stick.
        self._focus.secondary = focus
        self._focus.primary = self._pick_next_focus()
        n_nodes = len(nbhd["nodes"])
        return f"explored {focus}; {n_nodes} neighbor(s) in view", 0.05 + 0.01 * n_nodes

    def _compose(self) -> tuple[str, float]:
        focus = self._focus.primary
        secondary = self._focus.secondary or self._pick_next_focus()
        if not focus or not secondary or focus == secondary:
            return "no two concepts to compose", -0.05
        if not (self.universe.has(focus) and self.universe.has(secondary)):
            return "compose: missing concept", -0.05
        # Propose a child relation tying both into a candidate parent kind.
        composed_name = f"composed_{focus}_{secondary}"[:64]
        if self.universe.has(composed_name):
            return f"{composed_name} already exists", 0.0
        self.universe.add_concept(
            composed_name,
            domain="derived",
            definition=(
                f"Candidate parent kind inferred during composition of "
                f"{focus} and {secondary}."
            ),
            derived_from=(focus, secondary),
            salience=0.6,
        )
        for child in (focus, secondary):
            try:
                self.universe.add_relation(child, composed_name, "is_a", weight=0.6)
            except KeyError:
                continue
        return f"composed {focus} + {secondary} → {composed_name}", 0.25

    def _generalize(self) -> tuple[str, float]:
        focus = self._focus.primary
        if not focus or not self.universe.has(focus):
            return "no focus to generalize", -0.05
        concept = self.universe.expect(focus)
        # Find an existing concept the focus is_a or could be_a — if none,
        # propose one.
        is_a_neighbors = self.universe.neighbors(focus, kinds=["is_a"])
        if is_a_neighbors:
            target = is_a_neighbors[0].target
            self._focus.primary = target
            return f"generalized {focus} → {target}", 0.05
        parent = f"abstract_{focus}"
        if self.universe.has(parent):
            return f"{parent} already exists", 0.0
        self.universe.add_concept(
            parent,
            domain=concept.domain,
            definition=f"Abstract kind generalized from {focus}.",
            derived_from=(focus,),
            salience=0.5,
        )
        self.universe.add_relation(focus, parent, "is_a", weight=0.5)
        return f"generalized {focus} → {parent}", 0.2

    def _specialize(self) -> tuple[str, float]:
        focus = self._focus.primary
        if not focus or not self.universe.has(focus):
            return "no focus to specialize", -0.05
        concept = self.universe.expect(focus)
        # If something is_a focus, descend into it.
        children = [
            rel.source for rel in self.universe.neighbors(focus, include_incoming=True)
            if rel.target == focus and rel.kind == "is_a"
        ]
        if children:
            target = self._rng.choice(children)
            self._focus.primary = target
            return f"specialized {focus} → {target}", 0.05
        child = f"instance_{focus}_{self._step}"
        self.universe.add_concept(
            child,
            domain=concept.domain,
            definition=f"Instance of {focus} proposed at step {self._step}.",
            derived_from=(focus,),
            salience=0.45,
        )
        self.universe.add_relation(child, focus, "is_a", weight=0.5)
        return f"specialized {focus} → {child}", 0.15

    def _analogize(self) -> tuple[str, float]:
        focus = self._focus.primary
        if not focus or not self.universe.has(focus):
            return "no focus to analogize", -0.05
        center = self.universe.expect(focus)
        # Find any concept in a different domain that shares >=2 relation
        # kinds with the focus.
        focus_kinds = {rel.kind for rel in self.universe.neighbors(focus)}
        if not focus_kinds:
            return "focus has no relations to analogize from", -0.05
        best_name = ""
        best_score = 0.0
        for candidate in self.universe.all_concepts():
            if candidate.name == focus or candidate.domain == center.domain:
                continue
            cand_kinds = {rel.kind for rel in self.universe.neighbors(candidate.name)}
            if not cand_kinds:
                continue
            overlap = len(focus_kinds & cand_kinds) / max(1, len(focus_kinds | cand_kinds))
            if overlap > best_score:
                best_score = overlap
                best_name = candidate.name
        if not best_name or best_score < 0.2:
            return f"no cross-domain analogue found for {focus}", 0.0
        # Record the analogy as a graph edge.
        try:
            self.universe.add_relation(
                focus, best_name, "analogous_to", weight=0.5 + best_score,
                notes=f"jaccard={best_score:.2f}",
            )
        except KeyError:
            return f"could not link {focus} to {best_name}", -0.05
        return f"analogized {focus} ↔ {best_name}", 0.15 + 0.1 * best_score

    def _reflect(self) -> tuple[str, float]:
        focus = self._focus.primary
        if not focus or not self.universe.has(focus):
            return "no focus to reflect on", -0.05
        concept = self.universe.expect(focus)
        n_neighbors = len(self.universe.neighbors(focus))
        defn = concept.definition or "no definition yet"
        return (
            f"reflecting on {focus}: {defn} ({n_neighbors} link(s))",
            0.02 + 0.005 * n_neighbors,
        )

    def _derive(self) -> tuple[str, float]:
        if self.deriver is None:
            return "no deriver attached", -0.05
        accepted = self.deriver.derive()
        self._derivations_seen += len(accepted)
        if not accepted:
            return "deriver produced no new concepts this pass", 0.0
        return (
            f"deriver added {len(accepted)} concept(s) via "
            f"{', '.join(sorted({c.pathway for c in accepted}))}",
            0.1 + 0.05 * len(accepted),
        )

    def _wander(self) -> tuple[str, float]:
        all_concepts = self.universe.all_concepts()
        if not all_concepts:
            return "universe empty", -0.1
        target = min(all_concepts, key=lambda c: c.visits)
        target.visits += 1
        self._focus.secondary = self._focus.primary
        self._focus.primary = target.name
        return f"wandered to under-visited {target.name}", 0.05
