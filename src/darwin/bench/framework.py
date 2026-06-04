"""Benchmark framework — task / suite / runner / scorecard / comparison."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


# Canonical category names. Tasks declare which category they belong to;
# scorecards aggregate per-category and overall.
CATEGORIES: tuple[str, ...] = (
    "coding",
    "memory",
    "learning",
    "adaptation",
    "planning",
    "reasoning",
    "task_completion",
    "capability",
)


@dataclass
class BenchmarkTask:
    """One scored evaluation.

    ``runner`` is a callable that receives the live ``DarwinRuntime`` and
    must return a 0..1 score plus an evidence dict the framework records.
    """

    task_id: str
    category: str
    description: str
    runner: Callable[[Any], tuple[float, dict[str, Any]]]
    weight: float = 1.0
    timeout_seconds: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Outcome of one task on one runtime."""

    task_id: str
    category: str
    description: str
    score: float
    weight: float
    duration_ms: float
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreCard:
    """Aggregated results for one suite run.

    ``per_category`` is a weighted average per category; ``overall`` is the
    weighted average across every task. ``label`` is a free-form tag
    (typically the Darwin instance / commit / day the run represents).
    """

    scorecard_id: str
    label: str
    started_at: float
    completed_at: float
    overall: float
    per_category: dict[str, float]
    results: list[BenchmarkResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "results": [r.to_record() for r in self.results],
        }


# --------------------------------------------------------------------------- #
# Suite
# --------------------------------------------------------------------------- #


@dataclass
class BenchmarkSuite:
    """A collection of BenchmarkTasks with a label."""

    name: str
    tasks: list[BenchmarkTask] = field(default_factory=list)

    def add(self, task: BenchmarkTask) -> None:
        if task.category not in CATEGORIES:
            raise ValueError(
                f"task category {task.category!r} not in {CATEGORIES}"
            )
        self.tasks.append(task)

    def by_category(self) -> dict[str, list[BenchmarkTask]]:
        out: dict[str, list[BenchmarkTask]] = {c: [] for c in CATEGORIES}
        for task in self.tasks:
            out[task.category].append(task)
        return out


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


class BenchmarkRunner:
    """Execute a BenchmarkSuite against a runtime and produce a ScoreCard."""

    def __init__(self, suite: BenchmarkSuite) -> None:
        self.suite = suite

    def run(self, runtime: Any, *, label: str = "") -> ScoreCard:
        started = time.time()
        results: list[BenchmarkResult] = []
        for task in self.suite.tasks:
            t0 = time.perf_counter()
            score = 0.0
            evidence: dict[str, Any] = {}
            error = ""
            try:
                outcome = task.runner(runtime)
                if isinstance(outcome, tuple) and len(outcome) == 2:
                    score, evidence = outcome
                elif isinstance(outcome, (int, float)):
                    score = float(outcome)
                else:
                    score = float(outcome or 0.0)
                score = max(0.0, min(1.0, float(score)))
                evidence = dict(evidence or {})
            except Exception as exc:
                score = 0.0
                error = f"{type(exc).__name__}: {exc}"
            duration_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)
            results.append(BenchmarkResult(
                task_id=task.task_id,
                category=task.category,
                description=task.description,
                score=score,
                weight=task.weight,
                duration_ms=duration_ms,
                evidence=evidence,
                error=error,
            ))
        completed = time.time()
        per_category = _aggregate_per_category(results)
        overall = _aggregate_overall(results)
        return ScoreCard(
            scorecard_id=uuid.uuid4().hex[:12],
            label=label or self.suite.name,
            started_at=started,
            completed_at=completed,
            overall=overall,
            per_category=per_category,
            results=results,
        )


def _aggregate_per_category(results: list[BenchmarkResult]) -> dict[str, float]:
    by_cat: dict[str, list[BenchmarkResult]] = {c: [] for c in CATEGORIES}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
    out: dict[str, float] = {}
    for cat, items in by_cat.items():
        if not items:
            continue
        total_weight = sum(r.weight for r in items)
        if total_weight <= 0:
            out[cat] = 0.0
            continue
        out[cat] = sum(r.score * r.weight for r in items) / total_weight
    return out


def _aggregate_overall(results: list[BenchmarkResult]) -> float:
    if not results:
        return 0.0
    total = sum(r.weight for r in results)
    if total <= 0:
        return 0.0
    return sum(r.score * r.weight for r in results) / total


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_scorecard(card: ScoreCard, path: str | Path) -> bool:
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(card.to_record(), handle, separators=(",", ":"))
        return True
    except OSError:
        return False


def load_scorecard(path: str | Path) -> ScoreCard | None:
    source = Path(path)
    if not source.exists():
        return None
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        results = [
            BenchmarkResult(**r) for r in payload.get("results", [])
            if isinstance(r, dict)
        ]
        return ScoreCard(
            scorecard_id=payload.get("scorecard_id", uuid.uuid4().hex[:12]),
            label=payload.get("label", ""),
            started_at=float(payload.get("started_at", 0.0)),
            completed_at=float(payload.get("completed_at", 0.0)),
            overall=float(payload.get("overall", 0.0)),
            per_category=dict(payload.get("per_category", {})),
            results=results,
            metadata=dict(payload.get("metadata", {})),
        )
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


@dataclass
class CategoryDelta:
    category: str
    earlier: float
    later: float
    delta: float

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Comparison:
    earlier_label: str
    later_label: str
    earlier_overall: float
    later_overall: float
    overall_delta: float
    per_category: list[CategoryDelta]
    winner: str            # "later" / "earlier" / "tie"

    def to_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "per_category": [c.to_record() for c in self.per_category],
        }


def compare_scorecards(earlier: ScoreCard, later: ScoreCard) -> Comparison:
    overall_delta = later.overall - earlier.overall
    cats = set(earlier.per_category) | set(later.per_category)
    per_cat = []
    for c in sorted(cats):
        e = float(earlier.per_category.get(c, 0.0))
        l = float(later.per_category.get(c, 0.0))
        per_cat.append(CategoryDelta(category=c, earlier=e, later=l, delta=l - e))
    if overall_delta > 1e-6:
        winner = "later"
    elif overall_delta < -1e-6:
        winner = "earlier"
    else:
        winner = "tie"
    return Comparison(
        earlier_label=earlier.label,
        later_label=later.label,
        earlier_overall=earlier.overall,
        later_overall=later.overall,
        overall_delta=overall_delta,
        per_category=per_cat,
        winner=winner,
    )
