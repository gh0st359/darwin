"""Tests for the five-tier memory consolidation stack."""

from __future__ import annotations

from darwin.mysterio.memory_tiers import (
    ArchetypalMemoryTier,
    ConceptualMemoryTier,
    EpisodicMemoryTier,
    MemoryItem,
    MemoryTierStack,
    NarrativeMemoryTier,
    SemanticMemoryTier,
)
from darwin.types import Transition


def _transition(reward: float = 1.0, t: int = 0) -> Transition:
    return Transition(
        before={"x": 0},
        action="advance",
        after={"x": 1},
        reward=reward,
        t=t,
        metadata={"track": "grounded"},
    )


def test_episodic_tier_ingests_transition() -> None:
    tier = EpisodicMemoryTier()
    for i in range(5):
        tier.ingest(_transition(t=i), track="grounded")
    assert tier.size() == 5
    items = tier.by_track(track="grounded")
    assert len(items) == 5
    assert all(isinstance(item, MemoryItem) for item in items)


def test_semantic_tier_consolidates_from_episodic() -> None:
    episodic = EpisodicMemoryTier()
    for i in range(20):
        episodic.ingest(_transition(reward=1.0, t=i), track="grounded")
    semantic = SemanticMemoryTier()
    result = semantic.consolidate(episodic)
    assert semantic.size() > 0
    assert result["tier"] == "semantic"
    assert result["consolidated"] >= 1


def test_conceptual_tier_consolidates_from_semantic() -> None:
    semantic = SemanticMemoryTier()
    # The conceptual tier splits semantic text on "⇒" and groups by head;
    # it promotes any head shared by >=3 items.
    for i in range(4):
        semantic.add(MemoryItem(text=f"advance ⇒ x{i}", salience=2.5))
    semantic.add(MemoryItem(text="retreat ⇒ y", salience=1.0))
    conceptual = ConceptualMemoryTier()
    result = conceptual.consolidate(semantic)
    assert conceptual.size() > 0
    assert result["tier"] == "conceptual"
    assert result["consolidated"] >= 1


def test_archetypal_tier_consolidates_from_conceptual() -> None:
    conceptual = ConceptualMemoryTier()
    for i in range(6):
        conceptual.add(
            MemoryItem(text=f"concept-{i}", salience=2.0, metadata={"signature": "shared"})
        )
    archetypal = ArchetypalMemoryTier()
    result = archetypal.consolidate(conceptual)
    assert archetypal.size() > 0
    assert result["tier"] == "archetypal"


def test_narrative_tier_promotes_chunks() -> None:
    class _StubChunk:
        chunk_id = "chunk-1"
        text = "I have been thinking about the partition between grounded and interior."

    tier = NarrativeMemoryTier()
    result = tier.consolidate_narrative([_StubChunk()])
    assert result["consolidated"] == 1
    assert tier.size() == 1
    items = tier.all()
    assert items[0].track == "interior"


def test_memory_tier_stack_step_runs_full_consolidation() -> None:
    stack = MemoryTierStack()
    for i in range(40):
        stack.ingest_transition(_transition(reward=1.0, t=i), track="grounded")
    step = stack.step()
    # Each tier returns its own pass summary; the stack should aggregate.
    assert "semantic" in step
    assert "conceptual" in step
    assert "archetypal" in step
    assert stack.episodic.size() == 40
    # Semantic tier ingested SOME consolidated items.
    assert stack.semantic.size() >= 1
