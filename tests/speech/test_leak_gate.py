"""Tests for the LeakGate — chat output never leaks structured internals."""

from __future__ import annotations

from darwin.speech.leak_gate import LeakGate


def test_clean_prose_passes() -> None:
    gate = LeakGate()
    result = gate.check("A neuron is a cell. It carries electrical signals.")
    assert result.passed
    assert result.reasons == []


def test_curly_braces_rejected() -> None:
    gate = LeakGate()
    result = gate.check('Here is the result: {"answer": 42}')
    assert not result.passed
    assert any("curly" in r.lower() or "forbidden" in r.lower() for r in result.reasons)


def test_event_stream_marker_rejected() -> None:
    gate = LeakGate()
    result = gate.check("Some prose then [event mesh_firing] more prose.")
    assert not result.passed


def test_json_key_value_rejected() -> None:
    gate = LeakGate()
    result = gate.check('Response includes "thesis": and "answer_points":')
    assert not result.passed


def test_slash_command_at_line_start_rejected() -> None:
    gate = LeakGate()
    result = gate.check("First line\n/snapshot\nthen normal prose")
    assert not result.passed


def test_bus_topic_token_rejected() -> None:
    gate = LeakGate()
    result = gate.check("I'll publish to BusTopic.MESH_FIRING for the listeners.")
    assert not result.passed


def test_repr_fragment_rejected() -> None:
    gate = LeakGate()
    result = gate.check("My state is <DarwinRuntime object at 0xDEADBEEF>.")
    assert not result.passed


def test_payload_field_token_rejected() -> None:
    gate = LeakGate()
    result = gate.check("I had answer_points and uncertainty_levels in my plan.")
    assert not result.passed


def test_bracketed_inference_tag_rejected() -> None:
    gate = LeakGate()
    result = gate.check("[is_a_chain] dog is a mammal.")
    assert not result.passed


def test_sanitized_fallback_is_non_empty_when_gate_fails() -> None:
    gate = LeakGate()
    result = gate.check('Stuff and {"a": 1} stuff.')
    assert not result.passed
    assert result.sanitized_fallback


def test_explicit_fallback_text_preferred_when_supplied() -> None:
    gate = LeakGate()
    result = gate.check(
        '{"thesis": "structured"}',
        fallback_text="I had something to say but couldn't phrase it cleanly.",
    )
    assert not result.passed
    assert "phrase it cleanly" in result.sanitized_fallback


def test_non_string_input_rejected_safely() -> None:
    gate = LeakGate()
    result = gate.check(None)  # type: ignore[arg-type]
    assert not result.passed
    assert "non-string" in result.reasons[0]
