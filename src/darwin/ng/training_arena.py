from __future__ import annotations

import html
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from darwin.paths import data_dir


TRAINING_PROMPTS = [
    "Observe yourself. What are you attending to, and what should become stronger?",
    "Choose a frontier curriculum task and explain how it would train you.",
    "Inspect your living-system needs. What pressure should drive the next action?",
    "Select a strategic objective and turn it into a concrete campaign step.",
    "Imagine a failed self-improvement experiment. How would you repair it?",
    "Use your body schema: what tool-world affordance should you master next?",
    "Model the operator as a collaborator. What should you ask or report?",
    "Raise the benchmark ladder. What harder test should replace an easy win?",
]


def _bar(value: float, width: int = 18) -> str:
    value = max(0.0, min(1.0, value))
    filled = int(round(value * width))
    return "#" * filled + "." * (width - filled)


def _truncate(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _safe_get(record: dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    cur: Any = record
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@dataclass
class TrainingFrame:
    cycle: int
    prompt: str
    reply: str
    frontier_score: float
    awareness_index: float
    strategic_power_index: float
    viability_index: float
    dominant_need: str
    top_objective: str
    top_curriculum_task: str
    activated_goal_ids: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "prompt": self.prompt,
            "reply": self.reply,
            "frontier_score": round(self.frontier_score, 4),
            "awareness_index": round(self.awareness_index, 4),
            "strategic_power_index": round(self.strategic_power_index, 4),
            "viability_index": round(self.viability_index, 4),
            "dominant_need": self.dominant_need,
            "top_objective": self.top_objective,
            "top_curriculum_task": self.top_curriculum_task,
            "activated_goal_ids": list(self.activated_goal_ids),
        }

    def render_ansi(self, width: int = 96) -> str:
        line = "=" * min(width, 96)
        metrics = (
            f"frontier [{_bar(self.frontier_score)}] {self.frontier_score:.2f}  "
            f"awareness [{_bar(self.awareness_index)}] {self.awareness_index:.2f}  "
            f"strategy [{_bar(self.strategic_power_index)}] {self.strategic_power_index:.2f}  "
            f"life [{_bar(self.viability_index)}] {self.viability_index:.2f}"
        )
        activated = (
            ", ".join(self.activated_goal_ids)
            if self.activated_goal_ids else "(none this cycle)"
        )
        return "\n".join(
            [
                line,
                f"Darwin NG :: cycle {self.cycle}",
                metrics,
                f"prompt: {_truncate(self.prompt, width - 8)}",
                f"reply: {_truncate(self.reply, width - 7)}",
                f"dominant need: {self.dominant_need}",
                f"top objective: {_truncate(self.top_objective, width - 15)}",
                f"curriculum: {_truncate(self.top_curriculum_task, width - 12)}",
                f"activated goals: {activated}",
            ]
        )


@dataclass
class TrainingSession:
    session_id: str
    started_at: float
    completed_at: float
    frames: list[TrainingFrame]
    trace_path: Path
    html_path: Path

    def to_record(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "frames": [frame.to_record() for frame in self.frames],
            "trace_path": str(self.trace_path),
            "html_path": str(self.html_path),
        }


class NGTrainingArena:
    """Visible self-training harness for Darwin NG.

    Each cycle talks to Darwin, runs a fresh NG cycle, optionally promotes
    self-generated goals, emits a terminal frame, appends JSONL evidence, and
    renders a browser-readable HTML trace.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        out_dir: str | Path | None = None,
        prompts: list[str] | None = None,
    ) -> None:
        self.runtime = runtime
        self.out_dir = Path(out_dir) if out_dir is not None else data_dir() / "ng_training"
        self.prompts = prompts or list(TRAINING_PROMPTS)

    def run(
        self,
        *,
        cycles: int = 8,
        activate_every: int = 3,
        visual: bool = True,
        delay: float = 0.0,
    ) -> TrainingSession:
        started = time.time()
        session_id = f"ngtrain_{uuid.uuid4().hex[:10]}"
        session_dir = self.out_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        trace_path = session_dir / "trace.jsonl"
        html_path = session_dir / "index.html"
        frames: list[TrainingFrame] = []
        with trace_path.open("w", encoding="utf-8") as trace:
            for cycle in range(1, cycles + 1):
                prompt = self._prompt_for(cycle, frames)
                reply = self.runtime.chat(prompt)
                state = self.runtime.run_ng_cycle(prompt)
                record = state.to_record()
                activated: list[str] = []
                if activate_every > 0 and cycle % activate_every == 0:
                    report = self.runtime.activate_ng_autonomy(limit=1)
                    activated = list(report.get("ledger_goal_ids", []))
                frame = self._frame_from_record(
                    cycle=cycle,
                    prompt=prompt,
                    reply=reply,
                    record=record,
                    activated=activated,
                )
                frames.append(frame)
                trace.write(json.dumps(frame.to_record(), sort_keys=True) + "\n")
                trace.flush()
                if visual:
                    print(frame.render_ansi(), flush=True)
                if delay > 0:
                    time.sleep(delay)
        session = TrainingSession(
            session_id=session_id,
            started_at=started,
            completed_at=time.time(),
            frames=frames,
            trace_path=trace_path,
            html_path=html_path,
        )
        self._write_html(session)
        return session

    def _prompt_for(self, cycle: int, frames: list[TrainingFrame]) -> str:
        base = self.prompts[(cycle - 1) % len(self.prompts)]
        if not frames:
            return base
        previous = frames[-1]
        return (
            f"{base} Previous cycle: frontier={previous.frontier_score:.2f}, "
            f"awareness={previous.awareness_index:.2f}, "
            f"strategy={previous.strategic_power_index:.2f}, "
            f"dominant_need={previous.dominant_need}."
        )

    def _frame_from_record(
        self,
        *,
        cycle: int,
        prompt: str,
        reply: str,
        record: dict[str, Any],
        activated: list[str],
    ) -> TrainingFrame:
        objectives = _safe_get(record, ["strategic_cortex", "objectives"], []) or []
        curriculum_tasks = _safe_get(record, ["frontier_curriculum", "tasks"], []) or []
        top_objective = objectives[0].get("name", "(none)") if objectives else "(none)"
        top_task = curriculum_tasks[0].get("title", "(none)") if curriculum_tasks else "(none)"
        return TrainingFrame(
            cycle=cycle,
            prompt=prompt,
            reply=reply,
            frontier_score=float(_safe_get(record, ["power_metrics", "total_frontier_score"], 0.0) or 0.0),
            awareness_index=float(_safe_get(record, ["awareness_system", "awareness_index"], 0.0) or 0.0),
            strategic_power_index=float(_safe_get(record, ["strategic_cortex", "strategic_power_index"], 0.0) or 0.0),
            viability_index=float(_safe_get(record, ["living_system", "viability_index"], 0.0) or 0.0),
            dominant_need=str(_safe_get(record, ["living_system", "needs", "dominant_need"], "")),
            top_objective=str(top_objective),
            top_curriculum_task=str(top_task),
            activated_goal_ids=activated,
        )

    def _write_html(self, session: TrainingSession) -> None:
        rows = []
        for frame in session.frames:
            rows.append(
                f"""
                <section class="frame">
                  <h2>Cycle {frame.cycle}</h2>
                  <div class="metrics">
                    <span>frontier <b>{frame.frontier_score:.2f}</b></span>
                    <span>awareness <b>{frame.awareness_index:.2f}</b></span>
                    <span>strategic <b>{frame.strategic_power_index:.2f}</b></span>
                    <span>living <b>{frame.viability_index:.2f}</b></span>
                  </div>
                  <p><strong>Prompt</strong>: {html.escape(frame.prompt)}</p>
                  <p><strong>Darwin</strong>: {html.escape(frame.reply)}</p>
                  <p><strong>Dominant need</strong>: {html.escape(frame.dominant_need)}</p>
                  <p><strong>Top objective</strong>: {html.escape(frame.top_objective)}</p>
                  <p><strong>Curriculum</strong>: {html.escape(frame.top_curriculum_task)}</p>
                  <p><strong>Activated goals</strong>: {html.escape(", ".join(frame.activated_goal_ids) or "(none)")}</p>
                </section>
                """
            )
        document = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Darwin NG Training Arena</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #101214; color: #f4f7fb; }}
    header {{ padding: 28px 36px; background: #1b2027; border-bottom: 1px solid #39414f; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    main {{ padding: 24px 36px 48px; display: grid; gap: 18px; }}
    .frame {{ border: 1px solid #3b4554; background: #171b21; border-radius: 8px; padding: 18px; }}
    .metrics {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0 14px; }}
    .metrics span {{ background: #222936; border: 1px solid #465268; border-radius: 999px; padding: 6px 10px; }}
    p {{ line-height: 1.45; }}
  </style>
</head>
<body>
  <header>
    <h1>Darwin NG Training Arena</h1>
    <div>session {html.escape(session.session_id)} · cycles {len(session.frames)} · trace {html.escape(str(session.trace_path))}</div>
  </header>
  <main>
    {''.join(rows)}
  </main>
</body>
</html>
"""
        session.html_path.write_text(document, encoding="utf-8")
