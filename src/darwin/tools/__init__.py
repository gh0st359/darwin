"""Real-world tool harness — Darwin acting on actual systems.

Six sandboxed adapters expose real-world capabilities as v5 Actions:

  * :class:`FilesystemTool` — bounded read/write/list/remove/stat.
  * :class:`TerminalTool` — shell with timeout and deny-list.
  * :class:`CodeExecutionTool` — Python in a subprocess sandbox.
  * :class:`WebTool` — http/https fetch + HTML→text.
  * :class:`GitTool` — read-only git status/log/diff/show/branch.
  * :class:`DatabaseTool` — sandboxed sqlite read + write.

The :class:`ToolRegistry` collects tools and dispatches by action name;
:class:`ToolWorld` presents a registry as a World protocol implementation
so Darwin's existing planner / experiment loop / interior simulator can
choose tool actions exactly as they choose any other action;
:class:`AutonomousRunner` drives long-running goal-directed sessions on a
ToolWorld with budgets and predicates.
"""

from darwin.tools.autonomous import AutonomousRunner, AutonomousStep, AutonomousTask
from darwin.tools.intent import IntentMatch, detect_intents
from darwin.tools.base import (
    SandboxEscape,
    Tool,
    ToolError,
    ToolResult,
    resolve_sandboxed,
)
from darwin.tools.code_execution import CodeExecutionTool
from darwin.tools.database import DatabaseTool
from darwin.tools.filesystem import FilesystemTool
from darwin.tools.git import GitTool
from darwin.tools.registry import ToolRegistry
from darwin.tools.terminal import TerminalTool
from darwin.tools.web import WebTool
from darwin.tools.world import ToolWorld


__all__ = [
    "AutonomousRunner",
    "AutonomousStep",
    "AutonomousTask",
    "CodeExecutionTool",
    "DatabaseTool",
    "FilesystemTool",
    "GitTool",
    "IntentMatch",
    "detect_intents",
    "SandboxEscape",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolWorld",
    "TerminalTool",
    "WebTool",
    "resolve_sandboxed",
]
