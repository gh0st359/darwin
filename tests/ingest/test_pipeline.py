"""Tests for IngestPipeline — end-to-end fact → universe + mesh."""

from __future__ import annotations

from darwin.ingest.pipeline import IngestPipeline
from darwin.mesh import CorticalMesh
from darwin.mysterio.bus import BusTopic, CognitionBus
from darwin.universe import build_default_universe


def test_pipeline_text_lands_facts_in_universe() -> None:
    universe = build_default_universe()
    pipeline = IngestPipeline(universe=universe)
    before = universe.summary()["relations"]
    pipeline.ingest_text("A neuron is a cell. Rain causes flooding.")
    after = universe.summary()["relations"]
    assert after > before


def test_pipeline_html_handles_markup() -> None:
    universe = build_default_universe()
    pipeline = IngestPipeline(universe=universe)
    pipeline.ingest_html("<p>A widget is a gadget.</p>")
    assert universe.has("widget")
    assert universe.has("gadget")


def test_pipeline_dedupes_repeated_facts() -> None:
    universe = build_default_universe()
    pipeline = IngestPipeline(universe=universe)
    pipeline.ingest_text("A maple is a tree. A maple is a tree.")
    assert pipeline.stats.facts_skipped_dup >= 1


def test_pipeline_activates_mesh_cells() -> None:
    universe = build_default_universe()
    mesh = CorticalMesh()
    pipeline = IngestPipeline(universe=universe, mesh=mesh)
    pipeline.ingest_text("A photon is a particle.")
    # Both endpoints now exist as cells in the mesh and carry activation.
    photon = mesh.cell("photon")
    particle = mesh.cell("particle")
    assert photon is not None and particle is not None
    assert photon.activation > 0


def test_pipeline_emits_bus_events() -> None:
    universe = build_default_universe()
    bus = CognitionBus()
    fact_events: list = []
    bus.subscribe(BusTopic.FACT_EXTRACTED, fact_events.append)
    pipeline = IngestPipeline(universe=universe, bus=bus)
    pipeline.ingest_text("A neuron is a cell.")
    assert fact_events


def test_pipeline_stats_track_throughput() -> None:
    universe = build_default_universe()
    pipeline = IngestPipeline(universe=universe)
    pipeline.ingest_text("A is B. C is D. E is F.")
    stats = pipeline.stats.to_record()
    assert stats["facts_seen"] >= 3
    assert stats["facts_added"] >= 3
    assert "facts_per_hour" in stats


def test_pipeline_skips_invalid_self_loop_facts() -> None:
    universe = build_default_universe()
    pipeline = IngestPipeline(universe=universe)
    # Force-inject an invalid fact directly.
    from darwin.ingest.nl_parser import Fact

    pipeline.ingest_facts([Fact(subject="x", predicate="is_a", object="x")])
    assert pipeline.stats.facts_skipped_invalid >= 1


def test_pipeline_empty_text_safe() -> None:
    universe = build_default_universe()
    pipeline = IngestPipeline(universe=universe)
    pipeline.ingest_text("")
    assert pipeline.stats.facts_added == 0
