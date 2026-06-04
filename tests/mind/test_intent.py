"""Intent / MindReply data classes."""

from __future__ import annotations

from darwin.mind.intent import Intent, IntentKind, MindReply


def test_intent_default_is_decline_and_not_actionable():
    intent = Intent()
    assert intent.kind is IntentKind.DECLINE
    assert intent.is_actionable() is False


def test_intent_dialogue_is_not_actionable():
    intent = Intent(kind=IntentKind.DIALOGUE, confidence=0.1)
    assert intent.is_actionable() is False


def test_intent_compute_is_actionable():
    intent = Intent(kind=IntentKind.COMPUTE, confidence=0.5)
    assert intent.is_actionable() is True


def test_intent_to_record_strips_embedding():
    intent = Intent(
        kind=IntentKind.COMPUTE, confidence=0.7,
        faculties=["calculator"], embedding=[1.0, 2.0],
    )
    rec = intent.to_record()
    assert "embedding" not in rec
    assert rec["kind"] == "compute"


def test_mind_reply_record_does_not_include_text_body():
    reply = MindReply(text="hello world", intent_kind="compute", confidence=0.9)
    rec = reply.to_record()
    # Provenance only — text length, not the text itself.
    assert "text" not in rec
    assert rec["text_length"] == len("hello world")
