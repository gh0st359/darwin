"""DefeasibleReasoner — defaults with exceptions.

"Birds fly" is a default rule. "Penguins don't fly" is an exception that
preempts the default when the subject is a penguin. The reasoner stores
both as typed rules and consults the universe to decide which fires for
a given query.

Defaults are added as ``Rule(source, kind, target, default=True)``.
Exceptions are added as ``Rule(specific_concept, kind, target,
default=False, preempts=parent_default_id)``. When asked about a
specific concept that has an exception, the exception wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DefaultRule:
    """A defeasible default of the form 'X is/causes/requires Y by default'."""

    rule_id: str
    source: str
    target: str
    kind: str
    polarity: bool = True   # True = "X relates to Y"; False = "X does NOT relate to Y"
    weight: float = 0.7


@dataclass
class Exception_:
    """Names a default rule that the exception preempts."""

    rule_id: str
    source: str
    target: str
    kind: str
    polarity: bool = False  # exceptions typically flip polarity
    preempts: str = ""


@dataclass
class DefeasibleVerdict:
    """A single resolution: what was concluded and which rule fired."""

    subject: str
    target: str
    kind: str
    holds: bool
    rule_id: str
    via_exception: bool = False
    via_subkind: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "target": self.target,
            "kind": self.kind,
            "holds": self.holds,
            "rule_id": self.rule_id,
            "via_exception": self.via_exception,
            "via_subkind": self.via_subkind,
        }


class DefeasibleReasoner:
    """Reason with defaults that exceptions can override."""

    def __init__(self, universe: Any) -> None:
        self.universe = universe
        self._defaults: dict[str, DefaultRule] = {}
        self._exceptions: dict[str, Exception_] = {}
        self._next_id = 1

    def add_default(
        self, source: str, kind: str, target: str, *, polarity: bool = True,
        weight: float = 0.7,
    ) -> DefaultRule:
        rule_id = f"d{self._next_id}"
        self._next_id += 1
        rule = DefaultRule(
            rule_id=rule_id, source=source, kind=kind, target=target,
            polarity=polarity, weight=weight,
        )
        self._defaults[rule_id] = rule
        return rule

    def add_exception(
        self, source: str, kind: str, target: str, *,
        preempts: str, polarity: bool = False,
    ) -> Exception_:
        rule_id = f"e{self._next_id}"
        self._next_id += 1
        exc = Exception_(
            rule_id=rule_id, source=source, kind=kind, target=target,
            polarity=polarity, preempts=preempts,
        )
        self._exceptions[rule_id] = exc
        return exc

    def query(self, subject: str, kind: str, target: str) -> DefeasibleVerdict | None:
        """Decide whether (subject, kind, target) holds by default."""

        # Direct exception?
        for exc in self._exceptions.values():
            if exc.source == subject and exc.kind == kind and exc.target == target:
                return DefeasibleVerdict(
                    subject=subject, target=target, kind=kind,
                    holds=exc.polarity, rule_id=exc.rule_id, via_exception=True,
                )
        # Inherited default via super-kind? Walk is_a edges upward to find
        # a default that applies to a more general kind.
        ancestors = self._ancestors(subject)
        for ancestor in [subject] + ancestors:
            for default in self._defaults.values():
                if (default.source == ancestor and default.kind == kind
                        and default.target == target):
                    # Check whether subject (or any tighter ancestor)
                    # has an exception that preempts this default.
                    preempting = self._find_exception_for(
                        subject, kind, target, default.rule_id,
                    )
                    if preempting is not None:
                        return DefeasibleVerdict(
                            subject=subject, target=target, kind=kind,
                            holds=preempting.polarity, rule_id=preempting.rule_id,
                            via_exception=True, via_subkind=ancestor,
                        )
                    return DefeasibleVerdict(
                        subject=subject, target=target, kind=kind,
                        holds=default.polarity, rule_id=default.rule_id,
                        via_subkind=ancestor if ancestor != subject else "",
                    )
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "defaults": len(self._defaults),
            "exceptions": len(self._exceptions),
        }

    # -- helpers -------------------------------------------------------

    def _ancestors(self, subject: str) -> list[str]:
        """Walk is_a edges upward from subject. Bounded."""

        if self.universe is None or not self.universe.has(subject):
            return []
        out: list[str] = []
        seen: set[str] = {subject}
        frontier: list[str] = [subject]
        while frontier:
            current = frontier.pop()
            try:
                edges = self.universe.neighbors(current, kinds=["is_a"])
            except Exception:
                edges = []
            for edge in edges:
                if edge.target in seen:
                    continue
                seen.add(edge.target)
                out.append(edge.target)
                frontier.append(edge.target)
            if len(out) > 32:
                break
        return out

    def _find_exception_for(
        self, subject: str, kind: str, target: str, default_id: str,
    ) -> Exception_ | None:
        # An exception fires if it names the default rule explicitly OR if
        # it shares the (kind, target) for a more specific subject.
        subject_chain = [subject] + self._ancestors(subject)
        for exc in self._exceptions.values():
            if exc.kind != kind or exc.target != target:
                continue
            if exc.preempts == default_id:
                if exc.source in subject_chain:
                    return exc
        return None


__all__ = ["DefaultRule", "DefeasibleReasoner", "DefeasibleVerdict", "Exception_"]
