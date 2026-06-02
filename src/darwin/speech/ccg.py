"""Compact Combinatory Categorial Grammar substrate.

CCG categories are either atomic (e.g. ``N``, ``S``, ``NP``) or
functional (e.g. ``S/NP`` — "needs an NP on the right to become an S";
``S\\NP`` — "needs an NP on the left to become an S"). The combinators
that compose them are:

  * Forward application:  ``X/Y, Y → X``
  * Backward application: ``Y, X\\Y → X``
  * Forward composition:  ``X/Y, Y/Z → X/Z``
  * Backward composition: ``Y\\Z, X\\Y → X\\Z``
  * Type-raising:         ``X → T/(T\\X)`` or ``X → T\\(T/X)``

This module provides a minimal but correct implementation. The
production pipeline uses it to *generate* sentences (the reverse of
parsing): given a content plan (list of typed lexical entries), walk
the grammar to produce a surface string.

The grammar is deliberately small. The lexicon grows by observation —
seeded from chat + reading via V-Ingest and (eventually) from the
operator's own utterances. New entries are added on the fly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CCGCategory:
    """A CCG category — atomic or functional."""

    atom: str = ""                # for atomic categories
    arg: "CCGCategory | None" = None
    result: "CCGCategory | None" = None
    direction: str = ""           # "/" or "\\" for functional; "" for atomic

    @classmethod
    def atomic(cls, name: str) -> "CCGCategory":
        return cls(atom=name)

    @classmethod
    def forward(cls, result: "CCGCategory", arg: "CCGCategory") -> "CCGCategory":
        return cls(arg=arg, result=result, direction="/")

    @classmethod
    def backward(cls, result: "CCGCategory", arg: "CCGCategory") -> "CCGCategory":
        return cls(arg=arg, result=result, direction="\\")

    @property
    def is_atomic(self) -> bool:
        return self.direction == ""

    def __str__(self) -> str:
        if self.is_atomic:
            return self.atom
        # Functional categories: result direction arg.
        # Wrap functional sub-parts in parens for unambiguous reading.
        result_str = str(self.result) if self.result is not None else "?"
        arg_str = str(self.arg) if self.arg is not None else "?"
        if self.arg is not None and not self.arg.is_atomic:
            arg_str = f"({arg_str})"
        return f"{result_str}{self.direction}{arg_str}"


# Canonical atomic categories used by the production lexicon.
N = CCGCategory.atomic("N")          # noun
NP = CCGCategory.atomic("NP")        # noun phrase
S = CCGCategory.atomic("S")          # sentence
ADJ = CCGCategory.atomic("ADJ")      # adjective
PREP = CCGCategory.atomic("PREP")    # preposition


@dataclass
class CCGSign:
    """A signed CCG node: category + a surface string (possibly partial)."""

    category: CCGCategory
    surface: str
    # Whether the surface comes from a single lexical entry vs. a
    # composition (useful for the generator to insert spacing).
    composed: bool = False

    def __str__(self) -> str:
        return f"{self.surface!r}:{self.category}"


def forward_apply(left: CCGSign, right: CCGSign) -> CCGSign | None:
    """X/Y, Y → X. Returns None if the categories don't match."""

    if left.category.direction != "/":
        return None
    if left.category.arg is None or left.category.result is None:
        return None
    if str(left.category.arg) != str(right.category):
        return None
    surface = (left.surface + " " + right.surface).strip()
    return CCGSign(
        category=left.category.result,
        surface=surface,
        composed=True,
    )


def backward_apply(left: CCGSign, right: CCGSign) -> CCGSign | None:
    """Y, X\\Y → X. Returns None if the categories don't match."""

    if right.category.direction != "\\":
        return None
    if right.category.arg is None or right.category.result is None:
        return None
    if str(right.category.arg) != str(left.category):
        return None
    surface = (left.surface + " " + right.surface).strip()
    return CCGSign(
        category=right.category.result,
        surface=surface,
        composed=True,
    )


def forward_compose(left: CCGSign, right: CCGSign) -> CCGSign | None:
    """X/Y, Y/Z → X/Z."""

    lc = left.category
    rc = right.category
    if lc.direction != "/" or rc.direction != "/":
        return None
    if lc.arg is None or rc.result is None or lc.result is None or rc.arg is None:
        return None
    if str(lc.arg) != str(rc.result):
        return None
    new_cat = CCGCategory.forward(lc.result, rc.arg)
    return CCGSign(
        category=new_cat,
        surface=(left.surface + " " + right.surface).strip(),
        composed=True,
    )


def combine(left: CCGSign, right: CCGSign) -> CCGSign | None:
    """Try forward/backward apply, then forward composition. First win."""

    for combinator in (forward_apply, backward_apply, forward_compose):
        out = combinator(left, right)
        if out is not None:
            return out
    return None


def parse_category(text: str) -> CCGCategory:
    """Parse a CCG category string like ``S\\NP/(NP/N)`` into a structure.

    Handles the standard precedence: ``/`` and ``\\`` are left-associative
    and equal precedence. Parentheses group.
    """

    text = text.strip()
    if not text:
        return CCGCategory.atomic("")
    # Strip outermost balanced parens.
    while text.startswith("(") and text.endswith(")") and _balanced(text[1:-1]):
        text = text[1:-1].strip()
    # Find the rightmost top-level "/" or "\\".
    depth = 0
    cut = -1
    direction = ""
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
        elif depth == 0 and ch in ("/", "\\"):
            cut = i
            direction = ch
            break
    if cut == -1:
        return CCGCategory.atomic(text)
    result = parse_category(text[:cut])
    arg = parse_category(text[cut + 1:])
    return CCGCategory(
        arg=arg, result=result, direction=direction,
    )


def _balanced(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


__all__ = [
    "ADJ",
    "CCGCategory",
    "CCGSign",
    "N",
    "NP",
    "PREP",
    "S",
    "backward_apply",
    "combine",
    "forward_apply",
    "forward_compose",
    "parse_category",
]
