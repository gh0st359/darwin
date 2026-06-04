"""V-Mind faculties — the cognitive sub-capabilities Darwin recruits internally.

Architecturally, faculties are what used to be ``darwin.agents``: code
synthesis, arithmetic, scientific reasoning, planning, research, and
dialogue. The difference is *dispatch*: the legacy ``AgentRegistry``
exposed each agent as a categorised persona that could surface in chat
("the code agent solved this"). The new :class:`Mind` recruits faculties
**internally** and composes a single coherent reply in Darwin's own
voice — no categorisation visible to the operator.

For one phase, the legacy class names (``CodeAgent``, ``MathAgent``,
``...``) and the ``AgentRegistry`` import path are preserved as
deprecation aliases so the existing test suite continues to import them.
New code should depend on :class:`Mind` (in :mod:`darwin.faculties.mind`)
not on individual faculties or on the registry.
"""

from __future__ import annotations

# Faculty classes are the legacy agent classes re-exported under the new
# semantic names. Faculties operate identically — only the dispatch
# surface above them has changed.
from darwin.agents.code_agent import CodeAgent as Coder, CodeProblem
from darwin.agents.dialogue_agent import (
    DialogueAgent as Conversationalist,
    DialogueProblem,
)
from darwin.agents.math_agent import MathAgent as Calculator, MathProblem
from darwin.agents.planning_agent import (
    PlanningAgent as Planner,
    PlanningProblem,
)
from darwin.agents.research_agent import (
    ResearchAgent as Researcher,
    ResearchProblem,
)
from darwin.agents.science_agent import (
    ScienceAgent as Scientist,
    ScienceProblem,
)
from darwin.faculties.mind import Mind
from darwin.mind.intent import Intent, IntentKind, MindReply

__all__ = [
    "Calculator",
    "Coder",
    "CodeProblem",
    "Conversationalist",
    "DialogueProblem",
    "Intent",
    "IntentKind",
    "Mind",
    "MindReply",
    "MathProblem",
    "Planner",
    "PlanningProblem",
    "Researcher",
    "ResearchProblem",
    "Scientist",
    "ScienceProblem",
]
