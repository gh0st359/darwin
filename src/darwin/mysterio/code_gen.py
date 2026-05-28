"""Code-level self-modification: Darwin writes new Python modules of itself.

This is the leap from scalar-tweak self-modification to *structural* growth.
A `MODULE` or `SUBSYSTEM` proposal carries a `ProposalSpec`; the
`CodeGenerator` turns that spec into a real Python source file under
`src/darwin/generated/`, the `ModuleLoader` imports it live and hooks it into
a registry, and rollback removes the file + unhooks the subsystem.

Two synthesis paths:

  1. Spec-supplied source — `spec.generated_code` is a literal module body the
     meta-proposer emitted. We parse it (``ast.parse``) to guarantee it is
     syntactically valid before it ever touches disk; a `ContainmentError` is
     raised otherwise so the bad module never bricks the run.

  2. Template synthesis — when no literal source is supplied, we emit one from
     an AST template selected by ``spec.extra["template"]``. Templates are
     parameterized by the observed regularity that motivated the proposal
     (a prediction-failure variable, a starved loop, a consolidation cadence).

Every generated module's full source is captured in the returned
`GeneratedModule` so the snapshot store can persist it; rollback restores
byte-equal state by deleting the file the generator created.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from darwin.mysterio.safety import ContainmentError, MutationKind
from darwin.mysterio.proposal_spec import ProposalSpec


def _package_root() -> Path:
    """Locate ``src/darwin`` so generated code lands inside the package."""
    return Path(__file__).resolve().parent.parent


GENERATED_PACKAGE = "darwin.generated"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sanitize_identifier(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")
    if not cleaned:
        cleaned = "mod"
    if cleaned[0].isdigit():
        cleaned = f"m_{cleaned}"
    return cleaned.lower()


@dataclass
class GeneratedModule:
    """Provenance record for a module Darwin wrote and loaded.

    ``source`` is the full text on disk; ``sha256`` lets the snapshot store
    detect drift and verify byte-equal rollback. ``loaded_object`` is the live
    module after import (``None`` until loaded).
    """

    name: str
    qualified_name: str
    path: Path
    source: str
    sha256: str
    spec_signature: str
    kind: MutationKind
    subsystem_name: str | None = None
    created_at: float = field(default_factory=time.time)
    loaded_object: Any = None

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "path": str(self.path),
            "sha256": self.sha256,
            "spec_signature": self.spec_signature,
            "kind": self.kind.value,
            "subsystem_name": self.subsystem_name,
            "created_at": self.created_at,
        }


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
#
# Each template returns a complete module source string. They are deliberately
# small and self-contained: a generated module exposes a known entrypoint
# (``build(context)``) that returns an object with the interface its kind
# implies. The loader calls ``build`` and registers the result.


def _template_sensor(spec: ProposalSpec) -> str:
    """A hypothesis-driven introspection sensor exposing a candidate factor."""
    factor = str(spec.extra.get("factor", "hidden_factor"))
    sensor_name = _sanitize_identifier(spec.extra.get("name", factor) + "_sensor")
    cls_name = "".join(part.capitalize() for part in sensor_name.split("_"))
    description = spec.description.replace('"""', "'''")
    return f'''"""Generated introspection sensor for candidate factor {factor!r}.

{description}

Darwin synthesized this module. It exposes the hidden factor {factor!r} as an
observable signal so the causal learner can test whether it explains a
standing prediction failure.
"""

from __future__ import annotations

from typing import Any


FACTOR = {factor!r}


class {cls_name}:
    """Side-effect-free probe over Darwin's state for factor {factor!r}."""

    factor = FACTOR

    def observe(self, darwin: Any) -> dict[str, Any]:
        self_model = getattr(darwin, "self_model", None)
        failures = dict(getattr(self_model, "prediction_failures", {{}}) or {{}})
        salience = sum(
            count for key, count in failures.items() if {factor!r} in key
        )
        return {{
            "factor": FACTOR,
            "salience": float(salience),
            "active": salience > 0,
        }}


def build(context: Any = None) -> {cls_name}:
    return {cls_name}()
'''


def _template_consolidator(spec: ProposalSpec) -> str:
    """A new memory-consolidation strategy at a chosen cadence."""
    cadence = float(spec.extra.get("cadence_seconds", 60.0))
    threshold = float(spec.extra.get("salience_threshold", 0.5))
    strat_name = _sanitize_identifier(spec.extra.get("name", "consolidator"))
    cls_name = "".join(part.capitalize() for part in strat_name.split("_")) or "Consolidator"
    description = spec.description.replace('"""', "'''")
    return f'''"""Generated consolidation strategy.

{description}

Darwin synthesized this consolidator. It scans episodic memory at a
{cadence:g}s cadence and promotes salient episodes above threshold
{threshold:g} into the semantic tier.
"""

from __future__ import annotations

from typing import Any


CADENCE_SECONDS = {cadence!r}
SALIENCE_THRESHOLD = {threshold!r}


class {cls_name}:
    cadence_seconds = CADENCE_SECONDS
    salience_threshold = SALIENCE_THRESHOLD

    def consolidate(self, darwin: Any) -> dict[str, Any]:
        memory = getattr(darwin, "memory", None)
        episodes = getattr(memory, "episodes", None)
        recent = list(episodes.recent(64)) if episodes is not None else []
        promoted = [
            t for t in recent
            if abs(float(getattr(t, "reward", 0.0))) >= SALIENCE_THRESHOLD
        ]
        return {{
            "scanned": len(recent),
            "promoted": len(promoted),
            "threshold": SALIENCE_THRESHOLD,
        }}


def build(context: Any = None) -> {cls_name}:
    return {cls_name}()
'''


def _template_subsystem(spec: ProposalSpec) -> str:
    """A new long-running cognitive subsystem with a tick() loop body."""
    sub_name = _sanitize_identifier(spec.extra.get("name", spec.target_module_path or "subsystem"))
    cls_name = "".join(part.capitalize() for part in sub_name.split("_")) or "Subsystem"
    topic = str(spec.extra.get("topic", "subsystem_event"))
    description = spec.description.replace('"""', "'''")
    return f'''"""Generated cognitive subsystem {sub_name!r}.

{description}

Darwin synthesized this subsystem. Its ``tick`` is invoked by the supervisor
each scheduling interval; it publishes to the {topic!r} bus topic.
"""

from __future__ import annotations

from typing import Any


SUBSYSTEM_NAME = {sub_name!r}
TOPIC = {topic!r}


class {cls_name}:
    name = SUBSYSTEM_NAME
    topic = TOPIC

    def __init__(self) -> None:
        self.tick_count = 0

    def tick(self, context: Any = None) -> dict[str, Any]:
        self.tick_count += 1
        return {{"subsystem": SUBSYSTEM_NAME, "tick": self.tick_count}}


def build(context: Any = None) -> {cls_name}:
    return {cls_name}()
'''


_TEMPLATES: dict[str, Callable[[ProposalSpec], str]] = {
    "sensor": _template_sensor,
    "consolidator": _template_consolidator,
    "subsystem": _template_subsystem,
}


class CodeGenerator:
    """Synthesizes Python modules from `MODULE`/`SUBSYSTEM` proposal specs.

    The generator is the only component permitted to write under
    ``src/darwin/generated/``. It guarantees every module it emits parses
    cleanly before hitting disk; otherwise it raises `ContainmentError` so a
    malformed synthesis can never load.
    """

    def __init__(self, generated_root: Path | str | None = None) -> None:
        if generated_root is None:
            generated_root = _package_root() / "generated"
        self.generated_root = Path(generated_root)
        self.generated_root.mkdir(parents=True, exist_ok=True)
        self._ensure_package_init(self.generated_root)
        self._registry: dict[str, GeneratedModule] = {}

    # -- synthesis ----------------------------------------------------------- #

    def synthesize(self, spec: ProposalSpec) -> GeneratedModule:
        if spec.kind not in (MutationKind.MODULE, MutationKind.SUBSYSTEM):
            raise ContainmentError(
                f"CodeGenerator only handles MODULE/SUBSYSTEM specs, got {spec.kind}"
            )
        source = self._render_source(spec)
        # Validate before anything else — a syntax error must not reach disk.
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise ContainmentError(f"generated source failed to parse: {exc!r}") from exc

        module_stem = self._module_stem(spec)
        qualified = f"{GENERATED_PACKAGE}.{module_stem}"
        path = self.generated_root / f"{module_stem}.py"
        subsystem_name = None
        if spec.kind is MutationKind.SUBSYSTEM:
            subsystem_name = _sanitize_identifier(
                spec.extra.get("name", module_stem)
            )
        return GeneratedModule(
            name=module_stem,
            qualified_name=qualified,
            path=path,
            source=source,
            sha256=_sha256(source),
            spec_signature=spec.introspection_signature,
            kind=spec.kind,
            subsystem_name=subsystem_name,
        )

    def _render_source(self, spec: ProposalSpec) -> str:
        if spec.generated_code:
            return spec.generated_code
        template_name = str(spec.extra.get("template", "subsystem"))
        template = _TEMPLATES.get(template_name)
        if template is None:
            raise ContainmentError(
                f"unknown code-gen template {template_name!r}; "
                f"known: {sorted(_TEMPLATES)}"
            )
        return template(spec)

    def _module_stem(self, spec: ProposalSpec) -> str:
        if spec.target_module_path:
            stem = Path(spec.target_module_path).stem
        else:
            stem = spec.extra.get("name") or spec.description[:24]
        base = _sanitize_identifier(stem)
        # Disambiguate by a short signature suffix so re-synthesis of the same
        # logical module under changed conditions doesn't clobber the prior file.
        suffix = spec.introspection_signature[:8]
        return f"{base}_{suffix}"

    # -- disk + load --------------------------------------------------------- #

    def write(self, module: GeneratedModule) -> Path:
        module.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_package_init(module.path.parent)
        module.path.write_text(module.source)
        self._registry[module.qualified_name] = module
        return module.path

    def _ensure_package_init(self, directory: Path) -> None:
        init_path = directory / "__init__.py"
        if not init_path.exists():
            init_path.write_text(
                '"""Darwin self-generated code. Tracked by SHA in the snapshot '
                'ledger; file contents are gitignored."""\n'
            )

    def registry(self) -> dict[str, GeneratedModule]:
        return dict(self._registry)

    def manifest(self) -> dict[str, str]:
        """path → sha256 manifest for the snapshot store."""
        return {str(m.path): m.sha256 for m in self._registry.values()}


class ModuleLoader:
    """Dynamic import, hot-swap, and rollback of generated modules.

    Loading invalidates the importlib caches so a freshly-written file is
    seen, imports it, calls its ``build(context)`` entrypoint, and records the
    built object in a registry keyed by qualified name. Rollback unimports the
    module, removes the file, and forgets the registry entry — restoring the
    on-disk state the snapshot captured.
    """

    def __init__(self, generator: CodeGenerator | None = None) -> None:
        self.generator = generator or CodeGenerator()
        self.loaded: dict[str, Any] = {}
        self.built: dict[str, Any] = {}

    def load(self, module: GeneratedModule, context: Any = None) -> Any:
        if not module.path.exists():
            self.generator.write(module)
        importlib.invalidate_caches()
        spec = importlib.util.spec_from_file_location(
            module.qualified_name, module.path
        )
        if spec is None or spec.loader is None:
            raise ContainmentError(f"could not build import spec for {module.path}")
        py_module = importlib.util.module_from_spec(spec)
        sys.modules[module.qualified_name] = py_module
        try:
            spec.loader.exec_module(py_module)
        except Exception as exc:
            sys.modules.pop(module.qualified_name, None)
            raise ContainmentError(
                f"generated module {module.qualified_name} failed to import: {exc!r}"
            ) from exc
        module.loaded_object = py_module
        self.loaded[module.qualified_name] = module
        builder = getattr(py_module, "build", None)
        if callable(builder):
            self.built[module.qualified_name] = builder(context)
        return py_module

    def get_built(self, qualified_name: str) -> Any:
        return self.built.get(qualified_name)

    def rollback(self, module: GeneratedModule) -> bool:
        """Unhook and delete a generated module. Returns True if removed."""
        sys.modules.pop(module.qualified_name, None)
        self.loaded.pop(module.qualified_name, None)
        self.built.pop(module.qualified_name, None)
        self.generator._registry.pop(module.qualified_name, None)
        removed = False
        if module.path.exists():
            module.path.unlink()
            removed = True
        return removed

    def is_loaded(self, qualified_name: str) -> bool:
        return qualified_name in self.loaded
