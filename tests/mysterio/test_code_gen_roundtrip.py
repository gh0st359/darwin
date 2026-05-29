"""Tests for code-level self-modification.

The CodeGenerator turns a MODULE/SUBSYSTEM proposal into a real Python file
under ``src/darwin/generated/``; the ModuleLoader imports it live and tracks
the built object. Rollback removes the file and forgets the registry entry.

This is the leap from scalar-tweak self-modification to *structural* growth:
Darwin literally writes new modules of itself, and rollback can restore
byte-equal on-disk state.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from darwin.mysterio.code_gen import CodeGenerator, GENERATED_PACKAGE, ModuleLoader
from darwin.mysterio.proposal_spec import ProposalSpec
from darwin.mysterio.safety import ContainmentError, MutationKind


def _spec(
    *,
    kind: MutationKind = MutationKind.SUBSYSTEM,
    name: str = "test_subsystem",
    description: str = "a probe synthesized by the test",
    template: str = "subsystem",
    code: str | None = None,
    extra: dict | None = None,
) -> ProposalSpec:
    extra_payload = {"name": name, "template": template, **(extra or {})}
    return ProposalSpec(
        kind=kind,
        target_paths=[f"darwin/generated/{name}.py"],
        touches={f"darwin/generated/{name}.py"},
        description=description,
        expected_effect="generated under unit test",
        target_module_path=f"darwin/generated/{name}.py",
        generated_code=code or None,
        extra=extra_payload,
    )


def test_template_subsystem_writes_importable_module(tmp_path: Path) -> None:
    generator = CodeGenerator(generated_root=tmp_path)
    loader = ModuleLoader(generator)

    spec = _spec(name="probe_alpha")
    module = generator.synthesize(spec)
    generator.write(module)

    assert module.path.exists()
    assert module.path.read_text() == module.source
    assert module.sha256

    built = loader.load(module)
    assert built is module.loaded_object
    # Templates expose `build(context)` returning an instance whose tick()
    # increments a counter.
    instance = loader.get_built(module.qualified_name)
    assert instance is not None
    first = instance.tick()
    second = instance.tick()
    assert first["tick"] == 1
    assert second["tick"] == 2


def test_rollback_removes_file_and_forgets_registry(tmp_path: Path) -> None:
    generator = CodeGenerator(generated_root=tmp_path)
    loader = ModuleLoader(generator)

    spec = _spec(name="probe_beta")
    module = generator.synthesize(spec)
    generator.write(module)
    loader.load(module)

    assert module.path.exists()
    assert loader.is_loaded(module.qualified_name)

    removed = loader.rollback(module)
    assert removed
    assert not module.path.exists()
    assert not loader.is_loaded(module.qualified_name)


def test_syntax_error_in_generated_source_is_quarantined(tmp_path: Path) -> None:
    generator = CodeGenerator(generated_root=tmp_path)
    spec = _spec(
        kind=MutationKind.MODULE,
        name="busted",
        code="def broken(:\n    pass\n",
    )
    with pytest.raises(ContainmentError):
        generator.synthesize(spec)
    # Nothing should have made it onto disk.
    assert list(tmp_path.glob("busted*.py")) == []


def test_unknown_template_rejected(tmp_path: Path) -> None:
    generator = CodeGenerator(generated_root=tmp_path)
    spec = _spec(template="nonexistent_template")
    with pytest.raises(ContainmentError):
        generator.synthesize(spec)


def test_manifest_lists_sha_per_written_module(tmp_path: Path) -> None:
    generator = CodeGenerator(generated_root=tmp_path)
    spec_a = _spec(name="probe_one")
    spec_b = _spec(name="probe_two")
    module_a = generator.synthesize(spec_a)
    module_b = generator.synthesize(spec_b)
    generator.write(module_a)
    generator.write(module_b)
    manifest = generator.manifest()
    assert str(module_a.path) in manifest
    assert manifest[str(module_a.path)] == module_a.sha256
    assert manifest[str(module_b.path)] == module_b.sha256


def test_sensor_template_observes_darwin_safely(tmp_path: Path) -> None:
    generator = CodeGenerator(generated_root=tmp_path)
    loader = ModuleLoader(generator)
    spec = _spec(
        kind=MutationKind.MODULE,
        name="probe_sensor",
        template="sensor",
        extra={"factor": "battery_charge"},
    )
    module = generator.synthesize(spec)
    generator.write(module)
    loader.load(module)
    sensor = loader.get_built(module.qualified_name)
    assert sensor is not None

    class _StubDarwin:
        class _StubSelfModel:
            prediction_failures = {"flip_switch:battery_charge": 3}

        self_model = _StubSelfModel()

    result = sensor.observe(_StubDarwin())
    assert result["factor"] == "battery_charge"
    assert result["active"] is True
    assert result["salience"] >= 3


def test_qualified_name_uses_generated_package(tmp_path: Path) -> None:
    generator = CodeGenerator(generated_root=tmp_path)
    spec = _spec(name="probe_named")
    module = generator.synthesize(spec)
    assert module.qualified_name.startswith(GENERATED_PACKAGE + ".")
