"""Tests for the continuity & visibility selection-pressure terms."""

from __future__ import annotations

from darwin.mysterio.continuity import (
    ContinuityConfig,
    ContinuitySnapshot,
    continuity_term,
    score_proposal,
    visibility_term,
)


def _snapshot(
    *,
    tracked: int = 4,
    beliefs: int = 10,
    ledger: float = 1.0,
    subsystems: int = 12,
    generated: int = 0,
    interior: int = 0,
    throughput: float = 0.0,
) -> ContinuitySnapshot:
    return ContinuitySnapshot(
        tracked_variables=tracked,
        high_conf_beliefs=beliefs,
        ledger_growth_rate=ledger,
        subsystem_count=subsystems,
        generated_module_count=generated,
        private_belief_count=interior,
        probe_throughput=throughput,
    )


def test_continuity_term_positive_when_substrate_grows() -> None:
    before = _snapshot()
    after = _snapshot(beliefs=15, generated=2, subsystems=14)
    assert continuity_term(before, after) > 0


def test_continuity_term_zero_or_negative_when_substrate_shrinks() -> None:
    before = _snapshot(beliefs=20, generated=4)
    after = _snapshot(beliefs=10, generated=2)
    assert continuity_term(before, after) <= 0


def test_visibility_term_positive_when_probe_throughput_rises() -> None:
    before = _snapshot(throughput=0.0)
    after = _snapshot(throughput=10.0)
    assert visibility_term(before, after) > 0


def test_score_proposal_combines_terms_under_config() -> None:
    before = _snapshot()
    after = _snapshot(beliefs=15, generated=1, throughput=5.0)
    config = ContinuityConfig(lambda_continuity=1.0, lambda_visibility=0.5)
    score = score_proposal(
        improvement=0.1, before=before, after=after, config=config
    )
    assert isinstance(score, float)


def test_continuity_snapshot_from_runtime_handles_missing_pieces() -> None:
    class _StubDarwin:
        class _StubCausalModel:
            def known_variables(self) -> list[str]:
                return ["a", "b"]

            def beliefs(self, limit: int = 100) -> list:
                return []

            def total_observations(self) -> int:
                return 0

        causal_model = _StubCausalModel()
        tracks = None

    class _StubRuntime:
        store = None
        supervisor = None
        code_generator = None
        divergence_probe = None
        embedding_space = None
        darwin = _StubDarwin()

    snap = ContinuitySnapshot.from_runtime(_StubRuntime())
    assert isinstance(snap, ContinuitySnapshot)
    assert snap.tracked_variables >= 0
