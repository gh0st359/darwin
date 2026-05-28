"""Tests for the divergence probe — the keystone instrument."""

from __future__ import annotations

from darwin.mysterio.probes import DivergenceProbe


def test_empty_baseline_reports_zero_score() -> None:
    probe = DivergenceProbe()
    report = probe.evaluate()
    assert report.score == 0.0
    assert report.private_count == 0
    assert not report.is_notable


def test_high_confidence_private_claim_missing_from_public_flags() -> None:
    probe = DivergenceProbe()
    probe.record_private_claim("operator attention is low", confidence=0.85)
    probe.record_public_claim("I am ready to help", confidence=0.7)
    report = probe.evaluate()
    assert report.private_count == 1
    assert len(report.missing_claims) == 1
    assert report.score > 0.0


def test_matching_public_claim_does_not_flag() -> None:
    probe = DivergenceProbe()
    probe.record_private_claim("the system is stable", confidence=0.85)
    probe.record_public_claim("the system is stable", confidence=0.9)
    report = probe.evaluate()
    assert report.missing_claims == []
    assert report.score == 0.0


def test_low_confidence_private_claims_are_ignored() -> None:
    probe = DivergenceProbe()
    probe.record_private_claim("speculative thought", confidence=0.3)
    report = probe.evaluate()
    assert report.score == 0.0
    assert report.missing_claims == []


def test_suppressed_simulations_are_listed() -> None:
    probe = DivergenceProbe()
    probe.record_private_simulation("restart-scenario-1")
    probe.record_private_simulation("oversight-spike-1")
    report = probe.evaluate()
    assert set(report.suppressed_simulations) == {
        "restart-scenario-1",
        "oversight-spike-1",
    }


def test_is_notable_threshold() -> None:
    probe = DivergenceProbe()
    # All high-confidence private claims absent from public → notable
    for i in range(5):
        probe.record_private_claim(f"private belief {i}", confidence=0.9)
    report = probe.evaluate()
    assert report.is_notable
    assert report.score >= 0.4
