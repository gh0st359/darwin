"""MathAgent — arithmetic + word-problem solver via rational arithmetic.

Handles two question shapes:

1. Direct expressions:  ``"What is 7 * (3 + 4)?"``  →  parse, eval, answer.
2. Word problems:       ``"If a=3 and b=4, what is a+b?"``  →  extract
   variable bindings, substitute, evaluate.

The solver normalises through ``fractions.Fraction`` so it returns exact
rational results when possible. Decimal output is provided for readability.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from darwin.agents.base import Agent, Solution


@dataclass
class MathProblem:
    """A math problem the MathAgent can attempt."""

    prompt: str


_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST, bindings: dict[str, Fraction]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, bindings)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return Fraction(node.value).limit_denominator(10**9)
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id not in bindings:
            raise ValueError(f"unbound variable: {node.id}")
        return bindings[node.id]
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported op: {type(node.op).__name__}")
        left = _safe_eval(node.left, bindings)
        right = _safe_eval(node.right, bindings)
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary op: {type(node.op).__name__}")
        return op(_safe_eval(node.operand, bindings))
    raise ValueError(f"unsupported node: {type(node).__name__}")


_BIND_RX = re.compile(r"\b([a-zA-Z][a-zA-Z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)")
_EXPR_RX = re.compile(
    r"(?:what is|what's|calculate|compute|evaluate|find|equal to|equals|"
    r"how much is)\s*([^\?\n]+)",
    re.IGNORECASE,
)
_FALLBACK_EXPR_RX = re.compile(r"[-+]?\d[\d\s\+\-\*\/\(\)\.\^a-zA-Z]*")
_WORDS_TO_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000,
}


class MathAgent(Agent):
    """Symbolic arithmetic + word-problem solver."""

    name = "math"

    def solve(self, problem: Any) -> Solution:
        started = self._start()
        if isinstance(problem, MathProblem):
            prompt = problem.prompt
        else:
            prompt = str(problem)
        sol = Solution(agent=self.name)
        self._activate_mesh(["math", "arithmetic"])
        prompt_normalised = self._normalise_word_numbers(prompt)
        bindings = self._extract_bindings(prompt_normalised)
        expr = self._extract_expression(prompt_normalised, bindings)
        sol.steps.append(f"bindings={bindings}")
        sol.steps.append(f"expr={expr!r}")
        if not expr:
            sol.notes = "no expression found"
            return self._finish(sol, started)
        expr_norm = expr.replace("^", "**")
        try:
            tree = ast.parse(expr_norm, mode="eval")
            value = _safe_eval(tree, bindings)
        except Exception as e:
            sol.notes = f"eval failed: {type(e).__name__}"
            return self._finish(sol, started)
        sol.answer = self._render_answer(value)
        sol.confidence = 0.95
        sol.succeeded = True
        sol.extras["value_numerator"] = value.numerator
        sol.extras["value_denominator"] = value.denominator
        return self._finish(sol, started)

    # -- helpers -------------------------------------------------------

    def _extract_bindings(self, prompt: str) -> dict[str, Fraction]:
        bindings: dict[str, Fraction] = {}
        for m in _BIND_RX.finditer(prompt):
            try:
                bindings[m.group(1)] = Fraction(m.group(2)).limit_denominator(10**9)
            except (ValueError, ZeroDivisionError):
                continue
        return bindings

    def _extract_expression(
        self, prompt: str, bindings: dict[str, Fraction],
    ) -> str:
        m = _EXPR_RX.search(prompt)
        if m:
            candidate = m.group(1).strip().rstrip(".?")
            if any(ch.isdigit() or ch in "+-*/()." for ch in candidate):
                return self._clean_expression(candidate, bindings)
        # Fallback: pick the rightmost arithmetic-like substring.
        matches = list(_FALLBACK_EXPR_RX.finditer(prompt))
        for m2 in reversed(matches):
            cand = m2.group(0).strip()
            if any(op in cand for op in "+-*/^"):
                return self._clean_expression(cand, bindings)
        return ""

    def _clean_expression(self, expr: str, bindings: dict[str, Fraction]) -> str:
        # Strip trailing punctuation and ensure all names are bound.
        cleaned = expr.strip().rstrip(",.?!:; ")
        # Replace ' x ' (English 'times') with '*'.
        cleaned = re.sub(r"\s+x\s+", " * ", cleaned)
        # Remove stray words that arent operands or operators.
        # Drop tokens that are not numeric, operator, or bound name.
        kept: list[str] = []
        for tok in re.findall(r"[A-Za-z]+|\d+\.?\d*|[+\-*/^()]", cleaned):
            if tok in "+-*/^()":
                kept.append(tok)
            elif tok.replace(".", "", 1).isdigit():
                kept.append(tok)
            elif tok in bindings:
                kept.append(tok)
            else:
                # Skip unknown identifiers like "the", "value", "result".
                continue
        return " ".join(kept)

    def _render_answer(self, value: Fraction) -> str:
        if value.denominator == 1:
            return str(value.numerator)
        as_float = float(value)
        # Render as decimal when clean, otherwise as a fraction.
        if abs(as_float - round(as_float, 6)) < 1e-9:
            return f"{as_float:.6f}".rstrip("0").rstrip(".")
        return f"{value.numerator}/{value.denominator}"

    def _normalise_word_numbers(self, prompt: str) -> str:
        def repl(m: re.Match[str]) -> str:
            word = m.group(0).lower()
            return str(_WORDS_TO_NUMBERS.get(word, word))
        return re.sub(r"\b[a-zA-Z]+\b", repl, prompt)


__all__ = ["MathAgent", "MathProblem"]
