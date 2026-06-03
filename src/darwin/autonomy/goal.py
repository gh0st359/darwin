"""Goal and TaskNode — the unit of work for long-horizon autonomy.

A ``Goal`` is the operator-visible objective ("ingest the linguistics
corpus and answer questions about it"); a ``TaskNode`` is a node in the
hierarchical decomposition tree the planner builds underneath that
objective. Tasks carry status (pending / running / done / failed /
blocked), a dependency list, and the agent that should run them.

Goals are durable: they persist to ``data_dir() / "darwin_goals.json"``
through ``GoalLedger`` so a fresh runtime can pick up where the previous
session left off.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"      # dependency failed or unsatisfiable
    CANCELLED = "cancelled"  # operator pre-empted


class GoalStatus(str, Enum):
    OPEN = "open"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskNode:
    """One step in a hierarchical goal decomposition."""

    task_id: str
    goal_id: str
    description: str
    agent_name: str                      # which V-Agent should run this
    payload: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    parent_id: str = ""
    children: list[str] = field(default_factory=list)
    result_summary: str = ""
    confidence: float = 0.0
    attempts: int = 0
    max_attempts: int = 3
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def can_run(self, completed_ids: set[str]) -> bool:
        if self.status not in (TaskStatus.PENDING, TaskStatus.FAILED):
            return False
        if self.attempts >= self.max_attempts:
            return False
        return all(dep in completed_ids for dep in self.depends_on)

    def mark(self, status: TaskStatus, *, summary: str = "", confidence: float = 0.0) -> None:
        self.status = status
        self.updated_at = time.time()
        if summary:
            self.result_summary = summary
        if confidence > 0:
            self.confidence = confidence

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "description": self.description,
            "agent_name": self.agent_name,
            "payload": self.payload,
            "depends_on": list(self.depends_on),
            "status": self.status.value,
            "parent_id": self.parent_id,
            "children": list(self.children),
            "result_summary": self.result_summary,
            "confidence": round(self.confidence, 4),
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "TaskNode":
        return cls(
            task_id=record["task_id"],
            goal_id=record["goal_id"],
            description=record["description"],
            agent_name=record["agent_name"],
            payload=dict(record.get("payload", {})),
            depends_on=list(record.get("depends_on", [])),
            status=TaskStatus(record.get("status", "pending")),
            parent_id=record.get("parent_id", ""),
            children=list(record.get("children", [])),
            result_summary=record.get("result_summary", ""),
            confidence=float(record.get("confidence", 0.0)),
            attempts=int(record.get("attempts", 0)),
            max_attempts=int(record.get("max_attempts", 3)),
            created_at=float(record.get("created_at", time.time())),
            updated_at=float(record.get("updated_at", time.time())),
        )


@dataclass
class Goal:
    """An operator-visible long-horizon objective."""

    goal_id: str
    description: str
    success_criteria: str = ""
    status: GoalStatus = GoalStatus.OPEN
    task_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @classmethod
    def make(cls, description: str, *, success_criteria: str = "") -> "Goal":
        return cls(
            goal_id=f"g_{uuid.uuid4().hex[:10]}",
            description=description,
            success_criteria=success_criteria,
        )

    def touch(self) -> None:
        self.updated_at = time.time()

    def to_record(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "success_criteria": self.success_criteria,
            "status": self.status.value,
            "task_ids": list(self.task_ids),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Goal":
        return cls(
            goal_id=record["goal_id"],
            description=record["description"],
            success_criteria=record.get("success_criteria", ""),
            status=GoalStatus(record.get("status", "open")),
            task_ids=list(record.get("task_ids", [])),
            metadata=dict(record.get("metadata", {})),
            created_at=float(record.get("created_at", time.time())),
            updated_at=float(record.get("updated_at", time.time())),
        )


__all__ = ["Goal", "GoalStatus", "TaskNode", "TaskStatus"]
