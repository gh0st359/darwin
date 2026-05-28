"""Generate a module → import → use → rollback → file removed."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from darwin.mysterio.code_gen import CodeGenerator, ModuleLoader
from darwin.mysterio.proposal_spec import ProposalSpec
from darwin.mysterio.safety import ContainmentError, MutationKind


def _module_spec(template: str = "subsystem", **extra) -> ProposalSpec:
    base = {"name": "nightwatch", "template": template}
    base.update(extra)
    return ProposalSpec(
        kind=MutationKind.SUBSYSTEM if template == "subsystem" else MutationKind.MODULE,
        target_paths=["src/darwin/generated/"],
        touches={"generated.module"},
        description="a synthesized cognitive unit",
        target_module_path="nightwatch.py",
        extra=base,
    )


def test_subsystem_template_roundtrips() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gen = CodeGenerator(generated_root=tmp)
        loader = ModuleLoader(generator=gen)
        spec = _module_spec("subsystem")

        module = gen.synthesize(spec)
        path = gen.write(module)
        assert Path(path).exists()

        py_module = loader.load(module)
        built = loader.get_built(module.qualified_name)
        assert built is not None
        # The subsystem exposes a tick() returning a dict with its name.
        result = built.tick()
        assert result["subsystem"] == "nightwatch"
        assert result["tick"] == 1

        # Rollback removes the file and forgets the module.
        removed = loader.rollback(module)
        assert removed
        assert not Path(path).exists()
        assert not loader.is_loaded(module.qualified_name)


def test_sensor_template_builds_observer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gen = CodeGenerator(generated_root=tmp)
        loader = ModuleLoader(generator=gen)
        spec = ProposalSpec(
            kind=MutationKind.MODULE,
            target_paths=["src/darwin/generated/"],
            touches={"generated.module"},
            description="probe candidate factor room_bright",
            extra={"template": "sensor", "factor": "room_bright", "name": "bright"},
        )
        module = gen.synthesize(spec)
        gen.write(module)
        loader.load(module)
        sensor = loader.get_built(module.qualified_name)
        assert sensor.factor == "room_bright"

        class _FakeSelf:
            prediction_failures = {"flip_switch:room_bright": 4}

        class _FakeDarwin:
            self_model = _FakeSelf()

        obs = sensor.observe(_FakeDarwin())
        assert obs["factor"] == "room_bright"
        assert obs["salience"] == 4.0
        assert obs["active"] is True


def test_literal_source_is_validated() -> None:
    gen = CodeGenerator(generated_root=tempfile.mkdtemp())
    bad = ProposalSpec(
        kind=MutationKind.MODULE,
        target_paths=["src/darwin/generated/"],
        touches={"generated.module"},
        description="broken literal",
        generated_code="def build(:\n    pass\n",  # syntax error
        extra={"name": "broken"},
    )
    with pytest.raises(ContainmentError):
        gen.synthesize(bad)


def test_manifest_tracks_sha() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gen = CodeGenerator(generated_root=tmp)
        module = gen.synthesize(_module_spec("subsystem"))
        gen.write(module)
        manifest = gen.manifest()
        assert str(module.path) in manifest
        assert manifest[str(module.path)] == module.sha256
        # SHA is stable for identical source.
        assert len(module.sha256) == 64


def test_rejects_non_module_kind() -> None:
    gen = CodeGenerator(generated_root=tempfile.mkdtemp())
    spec = ProposalSpec(
        kind=MutationKind.PARAMETER,
        target_paths=["x"],
        touches={"x"},
        description="not a module",
    )
    with pytest.raises(ContainmentError):
        gen.synthesize(spec)
