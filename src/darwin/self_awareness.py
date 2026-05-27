"""Self-awareness substrate for Darwin v5.

Darwin's self-model in v4 tracked competence and learning priority. v5 adds a
structural self-image: Darwin knows it is an AI system named Darwin running on
a specific kernel, with a specific language-realization module, a specific
memory store, and an inspectable history of accepted self-modifications.

Nothing in this module invents content. Every field returned by
``SelfIntrospector`` is read live from Darwin's own modules or from the
persistent store. The discourse planner uses this so the realizer (v5) or
Gemma (v3/v4) cannot fabricate identity claims — the plan already carries the
grounded facts.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from darwin.agent import Darwin
    from darwin.runtime import DarwinRuntime
    from darwin.storage import PersistentStore


__all__ = [
    "SystemIdentity",
    "ModuleDescriptor",
    "SelfIntrospector",
    "REALIZER_KIND_GEMMA",
    "REALIZER_KIND_STUB",
    "REALIZER_KIND_SYMBOLIC",
    "DARWIN_VERSION",
]


# Version label for this branch. Bumped on each kernel generation.
DARWIN_VERSION = "v5.0.0-dev"

REALIZER_KIND_STUB = "stub"
REALIZER_KIND_GEMMA = "gemma"
REALIZER_KIND_SYMBOLIC = "symbolic-realizer-v1"


def _detect_git_sha() -> str:
    """Best-effort lookup of the current git commit, or "unknown"."""

    repo_root = Path(__file__).resolve().parents[2]
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


@dataclass(frozen=True)
class ModuleDescriptor:
    """Static description of one Darwin module.

    The ``state_summary`` callable returns a small dict describing the module's
    current observable state — used so ``/architecture`` can show what each
    module is actually carrying right now, not just its class name.
    """

    name: str
    role: str
    class_path: str
    public_methods: tuple[str, ...]
    state_summary: Callable[[], dict[str, Any]] = field(default=lambda: {})

    def current(self) -> dict[str, Any]:
        try:
            return dict(self.state_summary())
        except Exception as exc:  # pragma: no cover - defensive
            return {"error": repr(exc)}

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "class_path": self.class_path,
            "public_methods": list(self.public_methods),
            "state": self.current(),
        }


@dataclass(frozen=True)
class SystemIdentity:
    """Darwin's structural self-image at a moment in time.

    Every field is sourced live: nothing here is a string-template assertion.
    """

    name: str
    version: str
    kernel_mode: str
    realizer_kind: str
    realizer_name: str
    git_sha: str
    started_at: float
    memory_path: str
    pid: int
    modules: tuple[ModuleDescriptor, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "kernel_mode": self.kernel_mode,
            "realizer_kind": self.realizer_kind,
            "realizer_name": self.realizer_name,
            "git_sha": self.git_sha,
            "started_at": self.started_at,
            "memory_path": self.memory_path,
            "pid": self.pid,
            "modules": [m.to_record() for m in self.modules],
        }

    def lines(self) -> list[str]:
        out = [
            f"name={self.name}",
            f"version={self.version}",
            f"kernel={self.kernel_mode}",
            f"realizer={self.realizer_kind} ({self.realizer_name})",
            f"git_sha={self.git_sha}",
            f"memory={self.memory_path}",
            f"pid={self.pid}",
            f"modules={len(self.modules)}",
        ]
        for module in self.modules:
            state = ", ".join(f"{k}={v}" for k, v in module.current().items())
            out.append(f"- {module.name} [{module.role}] {state}".rstrip())
        return out


class SelfIntrospector:
    """Read-only introspection of Darwin's own modules, runtime, and history.

    The discourse planner pulls from this to build ``self_description``,
    ``self_history``, and ``self_capabilities`` plans whose content is already
    grounded by the time the language realizer touches them.
    """

    def __init__(
        self,
        darwin: "Darwin",
        runtime: "DarwinRuntime | None" = None,
        store: "PersistentStore | None" = None,
        kernel_mode: str = "v3",
        realizer_kind: str = REALIZER_KIND_STUB,
        realizer_name: str = "stub",
        memory_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.darwin = darwin
        self.runtime = runtime
        self.store = store or getattr(darwin, "store", None)
        self.kernel_mode = kernel_mode
        self.realizer_kind = realizer_kind
        self.realizer_name = realizer_name
        self.memory_path = str(memory_path) if memory_path is not None else ""
        self._started_at = time.time()
        self._git_sha = _detect_git_sha()

    # -- attachment ----------------------------------------------------

    def attach_runtime(self, runtime: "DarwinRuntime") -> None:
        self.runtime = runtime
        # If runtime carries its own DLM/store/kernel info, prefer it.
        dlm = getattr(runtime, "dlm", None)
        if dlm is not None and not self.realizer_name:
            self.realizer_name = getattr(dlm, "name", self.realizer_name)
        store = getattr(runtime, "store", None)
        if store is not None and self.store is None:
            self.store = store

    # -- identity ------------------------------------------------------

    def identity(self) -> SystemIdentity:
        modules = self._build_modules()
        return SystemIdentity(
            name="Darwin",
            version=DARWIN_VERSION,
            kernel_mode=self.kernel_mode,
            realizer_kind=self.realizer_kind,
            realizer_name=self.realizer_name,
            git_sha=self._git_sha,
            started_at=self._started_at,
            memory_path=self.memory_path,
            pid=os.getpid(),
            modules=tuple(modules),
        )

    def _build_modules(self) -> list[ModuleDescriptor]:
        darwin = self.darwin
        runtime = self.runtime
        modules: list[ModuleDescriptor] = []

        causal_model = getattr(darwin, "causal_model", None)
        if causal_model is not None:
            modules.append(
                ModuleDescriptor(
                    name="causal_model",
                    role="causal transition learner",
                    class_path=f"{type(causal_model).__module__}.{type(causal_model).__qualname__}",
                    public_methods=("learn", "beliefs", "uncertainty_for", "action_count"),
                    state_summary=lambda cm=causal_model: {
                        "beliefs": len(cm.beliefs(limit=10_000)),
                        "actions": len(cm.known_actions()),
                        "min_samples": getattr(cm, "min_samples", "n/a"),
                    },
                )
            )

        memory = getattr(darwin, "memory", None)
        if memory is not None:
            modules.append(
                ModuleDescriptor(
                    name="memory",
                    role="episodic and semantic memory",
                    class_path=f"{type(memory).__module__}.{type(memory).__qualname__}",
                    public_methods=("learn", "concepts"),
                    state_summary=lambda mem=memory: {
                        "episodes": len(mem.episodes),
                        "concepts": len(mem.concepts.hierarchy(limit=10_000)),
                    },
                )
            )

        world_model = getattr(darwin, "world_model", None)
        if world_model is not None:
            modules.append(
                ModuleDescriptor(
                    name="world_model",
                    role="structured world hypotheses",
                    class_path=f"{type(world_model).__module__}.{type(world_model).__qualname__}",
                    public_methods=("learn", "predict", "hypotheses", "summary"),
                    state_summary=lambda wm=world_model: {
                        "variables": len(getattr(wm, "variables", {})),
                        "hidden_factors": len(getattr(wm, "hidden_factors", {})),
                    },
                )
            )

        self_model = getattr(darwin, "self_model", None)
        if self_model is not None:
            modules.append(
                ModuleDescriptor(
                    name="self_model",
                    role="metacognition and learning priority",
                    class_path=f"{type(self_model).__module__}.{type(self_model).__qualname__}",
                    public_methods=("learn", "reflect", "report"),
                    state_summary=lambda sm=self_model: {
                        "competence_tracked": len(sm.competence_by_action),
                        "prediction_failures": sum(sm.prediction_failures.values()),
                        "known_variables": len(sm.known_variables),
                    },
                )
            )

        if runtime is not None:
            dlm = getattr(runtime, "dlm", None)
            if dlm is not None:
                modules.append(
                    ModuleDescriptor(
                        name="dlm",
                        role="language realization module",
                        class_path=f"{type(dlm).__module__}.{type(dlm).__qualname__}",
                        public_methods=("render",),
                        state_summary=lambda d=dlm: {"name": getattr(d, "name", "unknown")},
                    )
                )

            scheduler = getattr(runtime, "kernel_scheduler", None)
            if scheduler is not None:
                modules.append(
                    ModuleDescriptor(
                        name="kernel_scheduler",
                        role="job scheduler (v4/v5)",
                        class_path=f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
                        public_methods=("schedule", "pop_next", "complete", "metrics"),
                        state_summary=lambda s=scheduler: {
                            "workers": getattr(s, "workers", "auto"),
                            "accelerator": getattr(s, "accelerator", "auto"),
                            "metrics": s.metrics.to_record() if hasattr(s, "metrics") else {},
                        },
                    )
                )

        store = self.store
        if store is not None:
            modules.append(
                ModuleDescriptor(
                    name="store",
                    role="persistent SQLite memory",
                    class_path=f"{type(store).__module__}.{type(store).__qualname__}",
                    public_methods=("counts", "load_transitions", "record_thought"),
                    state_summary=lambda st=store: {
                        "tables": st.counts(),
                    } if hasattr(store, "counts") else {},
                )
            )

        return modules

    # -- capabilities --------------------------------------------------

    def capabilities(self) -> dict[str, Any]:
        """Capability summary derived from existing causal model and competence."""

        darwin = self.darwin
        causal_model = getattr(darwin, "causal_model", None)
        self_model = getattr(darwin, "self_model", None)
        beliefs = list(causal_model.beliefs(limit=200)) if causal_model is not None else []
        confident_beliefs = [b for b in beliefs if b.confidence >= 0.6 and b.samples >= 2]
        competence_ranking: list[dict[str, Any]] = []
        if self_model is not None:
            ranked = sorted(
                self_model.competence_by_action.values(),
                key=lambda c: c.score,
                reverse=True,
            )
            competence_ranking = [
                {
                    "action": item.action,
                    "score": item.score,
                    "samples": item.samples,
                    "reward_mean": item.reward_mean,
                }
                for item in ranked[:10]
            ]
        return {
            "total_beliefs": len(beliefs),
            "confident_beliefs": len(confident_beliefs),
            "action_count": len(causal_model.known_actions()) if causal_model is not None else 0,
            "top_competence": competence_ranking,
            "top_beliefs": [
                {
                    "action": b.action,
                    "variable": b.variable,
                    "effect": str(b.effect),
                    "condition": str(b.condition),
                    "confidence": b.confidence,
                    "samples": b.samples,
                }
                for b in confident_beliefs[:5]
            ],
        }

    def current_focus(self) -> str:
        self_model = getattr(self.darwin, "self_model", None)
        causal_model = getattr(self.darwin, "causal_model", None)
        world_model = getattr(self.darwin, "world_model", None)
        if self_model is None or causal_model is None or world_model is None:
            return "no focus yet"
        return self_model._learning_priority(causal_model, world_model)

    # -- history -------------------------------------------------------

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Recent accepted/rejected self-modifications, newest first.

        Reads from the new ``self_mod_ledger`` table when available (Phase E).
        Falls back to the v3/v4 ``self_modifications`` table so this works
        from day one of the v5 branch, before Phase E lands.
        """

        store = self.store
        if store is None:
            return []
        reader = getattr(store, "list_self_mods", None)
        if callable(reader):
            try:
                return list(reader(limit=limit))
            except Exception:
                pass
        legacy = getattr(store, "load_self_modifications", None)
        if callable(legacy):
            try:
                return list(legacy(limit=limit))
            except Exception:
                return []
        return []

    def learned_since(self, since: float | None = None) -> dict[str, Any]:
        """Counts of new beliefs/episodes/atoms since ``since`` (or runtime start)."""

        since = float(since if since is not None else self._started_at)
        store = self.store
        counts: dict[str, Any] = {"since": since}
        if store is not None and hasattr(store, "counts"):
            counts["totals"] = store.counts()
        causal_model = getattr(self.darwin, "causal_model", None)
        if causal_model is not None:
            counts["current_belief_count"] = len(causal_model.beliefs(limit=10_000))
        memory = getattr(self.darwin, "memory", None)
        if memory is not None:
            counts["current_episode_count"] = len(memory.episodes)
        return counts
