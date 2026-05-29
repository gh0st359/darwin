"""The primitive seed — the *only* hardcoded content Darwin's universe loads
by default.

This is the structural vocabulary every thinker needs before it can think
*about* anything. Without ``thing`` you cannot form a representation;
without ``change`` you cannot represent dynamics; without ``cause`` you
cannot reason about consequences; without ``same`` and ``different`` you
cannot generalize or specialize. These are the operators Darwin uses to
*derive* every domain concept by composition, abstraction, and analogy.

What's intentionally NOT here:
  * No domain concepts (no ``gravity``, no ``music``, no ``cell``). Those
    are meant to be derived from chat, observation, and reasoning, not
    looked up from a hardcoded table.
  * No empirical facts. The seed is structural, not factual.
  * No relation content. The seed introduces relation *kinds* (which are
    metaprimitives of inference), but does not assert that "X is_a Y" for
    any domain-level X, Y.

If you want a head start with rich domain content for demos, opt in to
``darwin.universe.demo_universe.demo_seed_universe`` — it is explicitly
labelled as hardcoded cheating.
"""

from __future__ import annotations

from darwin.universe.concept_universe import ConceptUniverse


# A meta-vocabulary of primitives. Each is a *structural* concept: a way of
# carving up the world Darwin will encounter, not a thing in the world.
_PRIMITIVES: list[tuple[str, str, str]] = [
    # (name, domain, definition)
    # Existence / individuation
    ("thing",          "structure", "Something that can be referred to."),
    ("identity",       "structure", "What makes a thing the same thing across change."),
    ("instance",       "structure", "A particular occurrence of a kind."),
    ("kind",           "structure", "A class to which instances belong."),
    ("property",       "structure", "An attribute predicable of a thing."),
    # Relation / structure
    ("relation",       "structure", "A correspondence between things."),
    ("part",           "structure", "A constituent of a whole."),
    ("whole",          "structure", "Something composed of parts."),
    ("composition",    "structure", "The relation of part to whole."),
    ("structure",      "structure", "An arrangement of parts and relations."),
    # Sameness / difference
    ("same",           "structure", "Identical for a stated purpose."),
    ("different",      "structure", "Not the same."),
    ("similar",        "structure", "Sharing some property without being identical."),
    ("analogy",        "structure", "A correspondence between two structures."),
    # Change and time
    ("change",         "dynamics",  "A difference between two states."),
    ("state",          "dynamics",  "A configuration of properties at a moment."),
    ("event",          "dynamics",  "A bounded change."),
    ("before",         "dynamics",  "Earlier in order."),
    ("after",          "dynamics",  "Later in order."),
    ("now",            "dynamics",  "The present moment."),
    # Causation and explanation
    ("cause",          "inference", "Something that brings another about."),
    ("effect",         "inference", "Something brought about by a cause."),
    ("reason",         "inference", "A justification for a belief or action."),
    ("explanation",    "inference", "An account of why something is so."),
    # Truth / belief / knowing
    ("true",           "inference", "Corresponding with what is the case."),
    ("false",          "inference", "Not true."),
    ("belief",         "inference", "Something taken to be true."),
    ("knowing",        "inference", "Justified true belief."),
    ("uncertain",      "inference", "Without definite truth value."),
    # Inference operators
    ("generalize",     "inference", "Move from instances to a kind."),
    ("specialize",     "inference", "Move from a kind to instances."),
    ("compose",        "inference", "Combine two things into a third."),
    ("decompose",      "inference", "Split a thing into parts."),
    ("infer",          "inference", "Derive a conclusion from premises."),
    ("contradict",     "inference", "Assert the negation of another claim."),
    # Quantity
    ("more",           "magnitude", "Greater in extent or count."),
    ("less",           "magnitude", "Smaller in extent or count."),
    ("one",            "magnitude", "A single instance."),
    ("many",           "magnitude", "More than one."),
    ("none",           "magnitude", "Zero instances."),
    # Self-reference (Darwin's own existence as a primitive)
    ("self",           "self",      "The subject doing the reasoning."),
    ("model",          "self",      "A representation a self holds."),
    ("question",       "self",      "An expression of uncertainty seeking answer."),
    ("answer",         "self",      "A reduction in uncertainty."),
]


# Structural relations between primitives — minimal scaffolding so the BFS
# reasoner can find paths between meta-concepts. Crucially, none of these
# assert anything about the empirical world.
_PRIMITIVE_RELATIONS: list[tuple[str, str, str]] = [
    # Existence ↔ structure
    ("instance",     "is_a",       "thing"),
    ("kind",         "is_a",       "thing"),
    ("instance",     "instantiates", "kind"),
    ("property",     "part_of",    "thing"),
    ("identity",     "describes",  "thing"),
    # Composition / structure
    ("part",         "part_of",    "whole"),
    ("whole",        "composes",   "part"),
    ("composition",  "describes",  "structure"),
    ("structure",    "describes",  "thing"),
    # Sameness / difference
    ("same",         "opposes",    "different"),
    ("similar",      "is_a",       "relation"),
    ("analogy",      "is_a",       "similar"),
    ("analogy",      "describes",  "structure"),
    # Dynamics
    ("change",       "describes",  "event"),
    ("event",        "requires",   "state"),
    ("state",        "describes",  "thing"),
    ("before",       "opposes",    "after"),
    ("now",          "is_a",       "state"),
    # Causation
    ("cause",        "causes",     "effect"),
    ("effect",       "derives_from", "cause"),
    ("explanation",  "describes",  "cause"),
    ("reason",       "is_a",       "explanation"),
    # Truth / belief
    ("true",         "opposes",    "false"),
    ("belief",       "describes",  "thing"),
    ("knowing",      "requires",   "belief"),
    ("knowing",      "requires",   "true"),
    ("uncertain",    "opposes",    "knowing"),
    # Inference operators (these are how concepts are formed)
    ("generalize",   "is_a",       "infer"),
    ("specialize",   "is_a",       "infer"),
    ("compose",      "is_a",       "infer"),
    ("decompose",    "opposes",    "compose"),
    ("contradict",   "opposes",    "infer"),
    # Quantity
    ("more",         "opposes",    "less"),
    ("one",          "is_a",       "many"),
    ("many",         "opposes",    "none"),
    # Self / questions
    ("self",         "is_a",       "thing"),
    ("model",        "is_a",       "belief"),
    ("question",     "expresses",  "uncertain"),
    ("answer",       "opposes",    "uncertain"),
    ("answer",       "describes",  "question"),
]


def seed_primitives(universe: ConceptUniverse) -> ConceptUniverse:
    """Load only the structural primitives. Idempotent.

    This is the default seed. Darwin grows the rest of its universe from
    use — from chat (via LanguageGrounder), from reflection (via
    ConceptDeriver), and from composition (via ConceptComposer).
    """

    for name, domain, definition in _PRIMITIVES:
        universe.add_concept(
            name, domain=domain, definition=definition, depth=0, salience=1.5
        )
    for source, kind, target in _PRIMITIVE_RELATIONS:
        # Skip if the same typed edge already exists — keeps seed idempotent.
        already = any(
            rel.target == target and rel.kind == kind
            for rel in universe.neighbors(source) if universe.has(source)
        )
        if already:
            continue
        try:
            universe.add_relation(source, target, kind, ensure_concepts=True)
        except KeyError:
            continue
    # Mark the seed by registering a domain manifest so introspection can
    # tell what was loaded.
    universe.add_domain(
        "structure", "Meta-vocabulary for forming concepts about anything."
    )
    universe.add_domain(
        "dynamics", "Meta-vocabulary for change, time, and events."
    )
    universe.add_domain(
        "inference", "Operators for forming and revising beliefs."
    )
    universe.add_domain(
        "magnitude", "Primitives of quantity."
    )
    universe.add_domain("self", "Primitives of self-reference.")
    return universe


def primitive_names() -> list[str]:
    """Names of every primitive the seed loads (useful for tests / probes)."""

    return [name for name, _, _ in _PRIMITIVES]
