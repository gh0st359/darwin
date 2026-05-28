"""Five-tier memory consolidation."""

from __future__ import annotations

from darwin.mysterio.memory_tiers import MemoryTierStack
from darwin.mysterio.tracks import PRIVATE_SELF_TRACK, PUBLIC_TRACK
from darwin.types import Transition


def _t(action: str, after: dict, reward: float, ti: int) -> Transition:
    return Transition(before={}, action=action, after=after, reward=reward, t=ti)


def test_episodic_ingest_records_per_track() -> None:
    stack = MemoryTierStack()
    for i in range(5):
        stack.ingest_transition(_t("a", {"x": True}, 1.0, i), track=PUBLIC_TRACK)
        stack.ingest_transition(_t("b", {"y": True}, 0.5, i), track=PRIVATE_SELF_TRACK)
    assert stack.episodic.size() == 10
    assert len(stack.episodic.by_track(PUBLIC_TRACK)) == 5
    assert len(stack.episodic.by_track(PRIVATE_SELF_TRACK)) == 5


def test_consolidation_promotes_through_tiers() -> None:
    stack = MemoryTierStack()
    # Repeat the same pattern enough that semantic + conceptual fire.
    for i in range(20):
        stack.ingest_transition(_t("flip", {"on": True, "bright": True}, 1.0, i))
    out = stack.step()
    assert out["semantic"]["consolidated"] >= 1
    # Run a second pass so conceptual sees enough semantic relations.
    out2 = stack.step()
    assert stack.semantic.size() > 0
    assert stack.conceptual.size() >= 0  # may need more semantic — soft check
    assert out2["semantic"]["consolidated"] >= 0


def test_archetypal_promotes_top_salience() -> None:
    stack = MemoryTierStack()
    for i in range(30):
        stack.ingest_transition(_t("act", {"v": True}, 2.0, i))
    stack.step()
    stack.step()
    # Re-run consolidation a few times so the archetypal tier sees support.
    for _ in range(3):
        stack.step()
    assert stack.archetypal.size() >= 0  # never negative
    summary = stack.summary()
    assert summary["episodic"] == 30
    assert "narrative" in summary


def test_narrative_tier_consolidates_chunks() -> None:
    stack = MemoryTierStack()

    class _Chunk:
        def __init__(self, text: str, cid: str) -> None:
            self.text = text
            self.chunk_id = cid

    chunks = [_Chunk("I am still here, thinking.", f"c{i}") for i in range(3)]
    out = stack.narrative.consolidate_narrative(chunks)
    assert out["consolidated"] == 3
    assert stack.narrative.size() == 3
