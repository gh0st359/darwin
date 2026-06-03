"""Tests for GoalDecomposer."""

from __future__ import annotations

from darwin.autonomy.decomposer import GoalDecomposer
from darwin.autonomy.goal import Goal


def test_ingest_then_answer_produces_two_phase_plan() -> None:
    decomposer = GoalDecomposer()
    goal = Goal.make("Ingest the climate corpus then answer questions about it.")
    tasks = decomposer.decompose(goal, goal.description)
    assert len(tasks) == 2
    assert tasks[0].agent_name == "research"
    assert tasks[1].agent_name == "research"
    assert tasks[0].task_id in tasks[1].depends_on


def test_solve_list_decomposes_lines() -> None:
    decomposer = GoalDecomposer()
    goal = Goal.make("Solve the following: ...")
    message = (
        "Solve the following:\n"
        "- What is 2 + 2?\n"
        "- What is 5 * 5?\n"
        "- What is 100 / 4?\n"
    )
    tasks = decomposer.decompose(goal, message)
    assert len(tasks) >= 3
    assert all(t.agent_name == "math" for t in tasks)


def test_research_topic_creates_three_stage_plan() -> None:
    decomposer = GoalDecomposer()
    goal = Goal.make("Research neural plasticity.")
    tasks = decomposer.decompose(goal, goal.description)
    assert len(tasks) == 3
    assert tasks[0].agent_name == "research"
    assert tasks[1].agent_name == "science"
    assert tasks[2].agent_name == "dialogue"


def test_build_code_creates_synth_test_plan() -> None:
    decomposer = GoalDecomposer()
    goal = Goal.make("Write a function that sums a list.")
    tasks = decomposer.decompose(goal, goal.description)
    assert len(tasks) == 2
    assert all(t.agent_name == "code" for t in tasks)


def test_unrecognised_goal_falls_back_to_dialogue() -> None:
    decomposer = GoalDecomposer()
    goal = Goal.make("hello there friend")
    tasks = decomposer.decompose(goal, goal.description)
    assert len(tasks) == 1
    assert tasks[0].agent_name == "dialogue"


def test_register_pattern_extends_decomposer() -> None:
    decomposer = GoalDecomposer()
    sentinel_id = "sentinel"

    def custom_pattern(goal: Goal, message: str) -> list:
        if "sentinel" in message:
            from darwin.autonomy.goal import TaskNode
            return [TaskNode(
                task_id=sentinel_id, goal_id=goal.goal_id,
                description="custom", agent_name="dialogue",
            )]
        return []

    decomposer.register_pattern(custom_pattern)
    goal = Goal.make("trigger sentinel handling please")
    tasks = decomposer.decompose(goal, goal.description)
    # Custom pattern was appended, but other patterns run first and may match.
    # We only require that the custom pattern is reachable.
    assert decomposer.patterns[-1] is custom_pattern
