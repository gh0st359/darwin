"""AgentRegistry — single facade over all cognitive agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from darwin.agents.base import Agent
from darwin.agents.code_agent import CodeAgent
from darwin.agents.dialogue_agent import DialogueAgent
from darwin.agents.math_agent import MathAgent
from darwin.agents.planning_agent import PlanningAgent
from darwin.agents.research_agent import ResearchAgent
from darwin.agents.science_agent import ScienceAgent


@dataclass
class AgentRegistry:
    """Holds one instance of each agent, parameterised by runtime."""

    runtime: Any = None
    code: CodeAgent | None = None
    math: MathAgent | None = None
    science: ScienceAgent | None = None
    planning: PlanningAgent | None = None
    research: ResearchAgent | None = None
    dialogue: DialogueAgent | None = None

    def __post_init__(self) -> None:
        if self.code is None:
            self.code = CodeAgent(self.runtime)
        if self.math is None:
            self.math = MathAgent(self.runtime)
        if self.science is None:
            self.science = ScienceAgent(self.runtime)
        if self.planning is None:
            self.planning = PlanningAgent(self.runtime)
        if self.research is None:
            self.research = ResearchAgent(self.runtime)
        if self.dialogue is None:
            self.dialogue = DialogueAgent(self.runtime)

    def all(self) -> list[Agent]:
        return [
            self.code, self.math, self.science,
            self.planning, self.research, self.dialogue,
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "agents": [a.name for a in self.all()],
            "count": 6,
        }


__all__ = ["AgentRegistry"]
