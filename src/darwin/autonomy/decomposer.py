"""GoalDecomposer — turn a free-text goal into a TaskNode tree.

Rule-based decomposition for the v1 substrate. Each rule inspects the
goal string and emits a list of ``TaskNode`` objects. The patterns
cover the most common long-horizon shapes:

* ``ingest + answer``  — ingest a corpus, then answer N questions about it.
* ``solve list``        — solve a list of problems serially.
* ``research topic``   — ingest from URLs/text, then synthesize a report.
* ``build code``       — generate a function, then test it.

When no rule fires the decomposer falls back to a single-task plan that
routes the entire prompt through DialogueAgent. Better decomposers can
be layered in by registering callbacks via ``register_pattern()``.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from darwin.autonomy.goal import Goal, TaskNode


_INGEST_RX = re.compile(
    r"\b(?:ingest|read|process|absorb)\b\s+(.+?)\s+(?:then|and|,)\s*"
    r"(?:answer|tell\s+me|report|summarize|explain)",
    re.IGNORECASE,
)
_SOLVE_LIST_RX = re.compile(
    r"\b(?:solve|compute|evaluate|work\s+out)\s+(?:these|the\s+following|all)\b",
    re.IGNORECASE,
)
_RESEARCH_RX = re.compile(
    r"\b(?:research|investigate|study|look\s+into)\b\s+(.+)",
    re.IGNORECASE,
)
_BUILD_CODE_RX = re.compile(
    r"\b(?:write|build|implement|create)\b\s+(?:a\s+)?(?:function|program|script)",
    re.IGNORECASE,
)


PatternCallback = Callable[[Goal, str], list[TaskNode]]


def _new_task_id() -> str:
    return f"t_{uuid.uuid4().hex[:10]}"


def _ingest_then_answer(goal: Goal, message: str) -> list[TaskNode]:
    m = _INGEST_RX.search(message)
    if m is None:
        return []
    corpus_descriptor = m.group(1).strip().rstrip(".")
    ingest_id = _new_task_id()
    answer_id = _new_task_id()
    return [
        TaskNode(
            task_id=ingest_id, goal_id=goal.goal_id,
            description=f"Ingest corpus: {corpus_descriptor}",
            agent_name="research",
            payload={"phase": "ingest", "corpus": corpus_descriptor},
        ),
        TaskNode(
            task_id=answer_id, goal_id=goal.goal_id,
            description=f"Answer questions about {corpus_descriptor}",
            agent_name="research",
            payload={"phase": "answer", "corpus": corpus_descriptor},
            depends_on=[ingest_id],
        ),
    ]


def _solve_list(goal: Goal, message: str) -> list[TaskNode]:
    if not _SOLVE_LIST_RX.search(message):
        return []
    # Look for newline-separated items.
    items = [
        line.strip(" -*•\t")
        for line in message.splitlines()
        if line.strip() and len(line.strip()) > 3 and not _SOLVE_LIST_RX.search(line)
    ]
    if not items:
        return []
    tasks: list[TaskNode] = []
    for idx, item in enumerate(items, start=1):
        tasks.append(TaskNode(
            task_id=_new_task_id(), goal_id=goal.goal_id,
            description=f"Solve item {idx}: {item[:80]}",
            agent_name="math" if any(c.isdigit() for c in item) else "research",
            payload={"problem": item, "index": idx},
        ))
    return tasks


def _research_topic(goal: Goal, message: str) -> list[TaskNode]:
    m = _RESEARCH_RX.search(message)
    if m is None:
        return []
    topic = m.group(1).strip().rstrip(".")
    ingest_id = _new_task_id()
    reason_id = _new_task_id()
    report_id = _new_task_id()
    return [
        TaskNode(
            task_id=ingest_id, goal_id=goal.goal_id,
            description=f"Ingest available facts about {topic}",
            agent_name="research",
            payload={"phase": "ingest", "topic": topic},
        ),
        TaskNode(
            task_id=reason_id, goal_id=goal.goal_id,
            description=f"Reason over facts about {topic}",
            agent_name="science",
            payload={"phase": "reason", "topic": topic},
            depends_on=[ingest_id],
        ),
        TaskNode(
            task_id=report_id, goal_id=goal.goal_id,
            description=f"Compose report on {topic}",
            agent_name="dialogue",
            payload={"phase": "report", "topic": topic},
            depends_on=[reason_id],
        ),
    ]


def _build_code(goal: Goal, message: str) -> list[TaskNode]:
    if not _BUILD_CODE_RX.search(message):
        return []
    synth_id = _new_task_id()
    test_id = _new_task_id()
    return [
        TaskNode(
            task_id=synth_id, goal_id=goal.goal_id,
            description="Generate candidate implementation",
            agent_name="code",
            payload={"phase": "synthesize", "prompt": message},
        ),
        TaskNode(
            task_id=test_id, goal_id=goal.goal_id,
            description="Verify implementation against examples",
            agent_name="code",
            payload={"phase": "test", "prompt": message},
            depends_on=[synth_id],
        ),
    ]


def _single_dialogue(goal: Goal, message: str) -> list[TaskNode]:
    return [
        TaskNode(
            task_id=_new_task_id(), goal_id=goal.goal_id,
            description=message[:80] or "respond",
            agent_name="dialogue",
            payload={"message": message},
        ),
    ]


_DEFAULT_PATTERNS: tuple[PatternCallback, ...] = (
    _ingest_then_answer,
    _solve_list,
    _research_topic,
    _build_code,
)


@dataclass
class GoalDecomposer:
    """Build a list of TaskNodes from a free-text goal."""

    patterns: list[PatternCallback] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.patterns:
            self.patterns = list(_DEFAULT_PATTERNS)

    def register_pattern(self, callback: PatternCallback) -> None:
        self.patterns.append(callback)

    def decompose(self, goal: Goal, message: str) -> list[TaskNode]:
        """Try each pattern; return the first non-empty plan or a dialogue fallback."""

        for pattern in self.patterns:
            tasks = pattern(goal, message)
            if tasks:
                return tasks
        return _single_dialogue(goal, message)


__all__ = ["GoalDecomposer", "PatternCallback"]
