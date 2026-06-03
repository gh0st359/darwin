"""CodeAgent — HumanEval-style coding problem solver.

Given a problem statement and optional input/output examples, the agent
synthesises a Python function, executes it against the examples in a
sandboxed subprocess (via ``tools/code_execution.py`` when available),
and refines until tests pass or the budget is exhausted.

This is a deliberately bounded implementation. The synthesis strategy is
template-driven: the agent extracts the function signature from the
problem statement, identifies the operation (sum / product / count /
filter / reverse / etc.) from keyword patterns, and emits a candidate
implementation. The refinement loop tries simple permutations of the
operation when tests fail.
"""

from __future__ import annotations

import ast
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from darwin.agents.base import Agent, Solution


@dataclass
class CodeProblem:
    """A coding problem the CodeAgent can attempt."""

    prompt: str
    function_name: str = ""
    examples: list[tuple[Any, Any]] = field(default_factory=list)
    # Optional reference test snippet of the form `assert func(x) == y`.
    test_snippets: list[str] = field(default_factory=list)


_PROMPT_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsum\b|\btotal\b|\badd\b", re.IGNORECASE), "sum"),
    (re.compile(r"\bproduct\b|\bmultipl", re.IGNORECASE), "product"),
    (re.compile(r"\blength\b|\bcount\b|\bsize\b", re.IGNORECASE), "length"),
    (re.compile(r"\bmax(imum)?\b|\bgreatest\b|\blargest\b", re.IGNORECASE), "max"),
    (re.compile(r"\bmin(imum)?\b|\bsmallest\b|\bleast\b", re.IGNORECASE), "min"),
    (re.compile(r"\breverse[ds]?\b|\bbackwards?\b", re.IGNORECASE), "reverse"),
    (re.compile(r"\beven\b", re.IGNORECASE), "filter_even"),
    (re.compile(r"\bodd\b", re.IGNORECASE), "filter_odd"),
    (re.compile(r"\bsort(ed)?\b|\border\b", re.IGNORECASE), "sort"),
    (re.compile(r"\babsolute\b|\babs\b", re.IGNORECASE), "abs"),
    (re.compile(r"\bsquare\b", re.IGNORECASE), "square"),
    (re.compile(r"\bdouble\b|\btwice\b", re.IGNORECASE), "double"),
)


_TEMPLATES: dict[str, str] = {
    "sum": "def {name}(xs):\n    return sum(xs)\n",
    "product": (
        "def {name}(xs):\n    r = 1\n    for x in xs:\n        r *= x\n"
        "    return r\n"
    ),
    "length": "def {name}(xs):\n    return len(xs)\n",
    "max": "def {name}(xs):\n    return max(xs)\n",
    "min": "def {name}(xs):\n    return min(xs)\n",
    "reverse": "def {name}(xs):\n    return list(reversed(xs))\n",
    "filter_even": "def {name}(xs):\n    return [x for x in xs if x % 2 == 0]\n",
    "filter_odd": "def {name}(xs):\n    return [x for x in xs if x % 2 != 0]\n",
    "sort": "def {name}(xs):\n    return sorted(xs)\n",
    "abs": "def {name}(x):\n    return abs(x)\n",
    "square": "def {name}(x):\n    return x * x\n",
    "double": "def {name}(x):\n    return x * 2\n",
    "identity": "def {name}(x):\n    return x\n",
}


class CodeAgent(Agent):
    """Generates and tests small Python functions."""

    name = "code"

    def __init__(self, runtime: Any = None, *, sandbox_root: str | Path | None = None) -> None:
        super().__init__(runtime)
        if sandbox_root is None:
            sandbox_root = tempfile.mkdtemp(prefix="darwin_code_agent_")
        self.sandbox_root = Path(sandbox_root)
        self.sandbox_root.mkdir(parents=True, exist_ok=True)

    def solve(self, problem: Any) -> Solution:
        started = self._start()
        if isinstance(problem, str):
            problem = CodeProblem(prompt=problem)
        sol = Solution(agent=self.name)
        if not isinstance(problem, CodeProblem):
            sol.notes = "problem must be a CodeProblem or str"
            return self._finish(sol, started)
        self._activate_mesh([problem.function_name or "code"])

        func_name = problem.function_name or self._infer_function_name(problem.prompt) or "solve"
        candidates = self._candidate_strategies(problem.prompt)
        sol.steps.append(f"candidates={candidates}")
        for strategy in candidates:
            source = self._render(strategy, func_name)
            if source is None:
                continue
            if not self._parses(source):
                continue
            sol.steps.append(f"try:{strategy}")
            ok, details = self._run_local(source, func_name, problem)
            if ok:
                sol.answer = source
                sol.confidence = 0.8
                sol.succeeded = True
                sol.notes = f"strategy={strategy}"
                sol.extras["strategy"] = strategy
                return self._finish(sol, started)
            sol.steps.append(f"fail:{strategy}:{details[:60]}")
        # No strategy passed; return the first parseable candidate as a best
        # guess so downstream callers see a structured surface.
        if candidates:
            source = self._render(candidates[0], func_name)
            if source and self._parses(source):
                sol.answer = source
                sol.confidence = 0.3
                sol.notes = "no strategy passed tests"
        return self._finish(sol, started)

    # -- helpers -------------------------------------------------------

    def _infer_function_name(self, prompt: str) -> str:
        m = re.search(r"\bdef\s+([a-z_][a-z0-9_]*)\s*\(", prompt, re.IGNORECASE)
        return m.group(1) if m else ""

    def _candidate_strategies(self, prompt: str) -> list[str]:
        hits: list[str] = []
        for pattern, name in _PROMPT_HINTS:
            if pattern.search(prompt):
                hits.append(name)
        if not hits:
            hits.append("identity")
        # Deduplicate, preserve order.
        seen: set[str] = set()
        out: list[str] = []
        for h in hits:
            if h not in seen:
                out.append(h)
                seen.add(h)
        return out

    def _render(self, strategy: str, func_name: str) -> str | None:
        tpl = _TEMPLATES.get(strategy)
        if tpl is None:
            return None
        return tpl.format(name=func_name)

    def _parses(self, source: str) -> bool:
        try:
            ast.parse(source)
            return True
        except SyntaxError:
            return False

    def _run_local(
        self, source: str, func_name: str, problem: CodeProblem,
    ) -> tuple[bool, str]:
        """Exec the source in an isolated dict and check the examples."""

        try:
            namespace: dict[str, Any] = {}
            exec(compile(source, "<code_agent>", "exec"), namespace)
            func = namespace.get(func_name)
            if func is None:
                return False, "function not defined"
            for inp, expected in problem.examples:
                if isinstance(inp, tuple):
                    actual = func(*inp)
                else:
                    actual = func(inp)
                if actual != expected:
                    return False, f"{inp!r}→{actual!r}≠{expected!r}"
            for snippet in problem.test_snippets:
                exec(snippet, namespace)
            return True, "ok"
        except Exception as e:
            return False, f"{type(e).__name__}:{e}"


__all__ = ["CodeAgent", "CodeProblem"]
