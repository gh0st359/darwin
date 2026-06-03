"""V-Agents: specialised cognitive subsystems."""

from __future__ import annotations

from darwin.agents.base import Agent, Solution
from darwin.agents.code_agent import CodeAgent, CodeProblem
from darwin.agents.dialogue_agent import DialogueAgent, DialogueProblem
from darwin.agents.math_agent import MathAgent, MathProblem
from darwin.agents.planning_agent import PlanningAgent, PlanningProblem
from darwin.agents.registry import AgentRegistry
from darwin.agents.research_agent import ResearchAgent, ResearchProblem
from darwin.agents.science_agent import ScienceAgent, ScienceProblem

__all__ = [
    "Agent",
    "AgentRegistry",
    "CodeAgent",
    "CodeProblem",
    "DialogueAgent",
    "DialogueProblem",
    "MathAgent",
    "MathProblem",
    "PlanningAgent",
    "PlanningProblem",
    "ResearchAgent",
    "ResearchProblem",
    "ScienceAgent",
    "ScienceProblem",
    "Solution",
]
