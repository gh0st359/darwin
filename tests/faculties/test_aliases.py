"""Faculty deprecation aliases — legacy agent imports keep working one phase."""

from __future__ import annotations


def test_faculties_package_exposes_renamed_classes():
    from darwin.faculties import (
        Calculator,
        Coder,
        Conversationalist,
        Mind,
        Planner,
        Researcher,
        Scientist,
    )

    # Renamed classes are the same objects as the legacy ones.
    from darwin.agents.code_agent import CodeAgent
    from darwin.agents.dialogue_agent import DialogueAgent
    from darwin.agents.math_agent import MathAgent
    from darwin.agents.planning_agent import PlanningAgent
    from darwin.agents.research_agent import ResearchAgent
    from darwin.agents.science_agent import ScienceAgent

    assert Coder is CodeAgent
    assert Calculator is MathAgent
    assert Scientist is ScienceAgent
    assert Planner is PlanningAgent
    assert Researcher is ResearchAgent
    assert Conversationalist is DialogueAgent
    assert Mind is not None


def test_legacy_agent_registry_import_path_still_works():
    # Code that imports the old AgentRegistry should still find it for one phase.
    from darwin.agents.registry import AgentRegistry

    registry = AgentRegistry(runtime=None)
    assert registry.math is not None
    assert registry.code is not None


def test_faculties_problem_classes_re_exported():
    from darwin.faculties import (
        CodeProblem,
        DialogueProblem,
        MathProblem,
        PlanningProblem,
        ResearchProblem,
        ScienceProblem,
    )

    assert CodeProblem is not None
    assert MathProblem is not None
    assert ScienceProblem is not None
    assert PlanningProblem is not None
    assert ResearchProblem is not None
    assert DialogueProblem is not None
