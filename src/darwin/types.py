from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

State = dict[str, Any]


@dataclass(frozen=True)
class Action:
    """An available intervention Darwin can perform."""

    name: str
    cost: float = 0.0
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Goal:
    """A desired world state plus planning weights."""

    desired: Mapping[str, Any]
    weights: Mapping[str, float] = field(default_factory=dict)
    reward_weight: float = 1.0
    progress_weight: float = 3.0
    exploration_weight: float = 0.2


@dataclass(frozen=True)
class Transition:
    """The core learning atom: before, intervention, after, and payoff."""

    before: Mapping[str, Any]
    action: str
    after: Mapping[str, Any]
    reward: float
    t: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

