from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


@dataclass
class PlanLogEntry:
    plan_id: str
    user_text: str
    semantic_summary: str
    plan: dict[str, Any]
    rendering: str
    critique: dict[str, Any]
    trace: dict[str, Any]
    renderer: str
    background: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "user_text": self.user_text,
            "semantic_summary": self.semantic_summary,
            "plan": self.plan,
            "rendering": self.rendering,
            "critique": self.critique,
            "trace": self.trace,
            "renderer": self.renderer,
            "background": self.background,
            "timestamp": self.timestamp,
        }


@dataclass
class BackgroundLogEntry:
    loop: str
    kind: str
    content: str
    payload: dict[str, Any]
    duration_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        return {
            "loop": self.loop,
            "kind": self.kind,
            "content": self.content,
            "payload": self.payload,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }


class StructuredLogger:
    """JSONL logger for ResponsePlans + background cognition.

    The plan log doubles as a corpus of (structured plan -> rendered text)
    pairs, which is the raw material for fine-tuning the DLM later.
    """

    def __init__(
        self,
        plan_log: Path | str | None = None,
        background_log: Path | str | None = None,
        metrics_log: Path | str | None = None,
        enabled: bool = True,
    ) -> None:
        from darwin.paths import (
            background_log_path,
            metrics_log_path,
            plan_log_path,
        )

        self.plan_log = Path(plan_log) if plan_log is not None else plan_log_path()
        self.background_log = (
            Path(background_log) if background_log is not None else background_log_path()
        )
        self.metrics_log = (
            Path(metrics_log) if metrics_log is not None else metrics_log_path()
        )
        self.enabled = enabled
        self._lock = threading.RLock()
        self.metrics: dict[str, float] = {
            "plans_logged": 0.0,
            "background_events": 0.0,
        }
        self._counters: dict[str, int] = {}
        if self.enabled:
            for path in (self.plan_log, self.background_log, self.metrics_log):
                path.parent.mkdir(parents=True, exist_ok=True)

    def log_plan(self, entry: PlanLogEntry) -> None:
        if not self.enabled:
            return
        with self._lock:
            with self.plan_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_record(), default=_json_default))
                handle.write("\n")
            self.metrics["plans_logged"] += 1.0

    def log_background(self, entry: BackgroundLogEntry) -> None:
        if not self.enabled:
            return
        with self._lock:
            with self.background_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_record(), default=_json_default))
                handle.write("\n")
            self.metrics["background_events"] += 1.0
            key = f"loop:{entry.loop}"
            self._counters[key] = self._counters.get(key, 0) + 1

    def log_metric(self, name: str, value: float, payload: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.metrics[name] = float(value)
            record = {
                "name": name,
                "value": float(value),
                "payload": dict(payload or {}),
                "timestamp": time.time(),
            }
            with self.metrics_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=_json_default))
                handle.write("\n")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "metrics": dict(self.metrics),
                "counters": dict(self._counters),
            }

    def read_plan_entries(self, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.plan_log.exists():
            return []
        with self.plan_log.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
        if limit is not None:
            lines = lines[-limit:]
        return [json.loads(line) for line in lines if line.strip()]
