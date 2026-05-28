"""Tests for the operator-tier event subscription."""

from __future__ import annotations

import os

import pytest

from darwin.mysterio.operator_channel import (
    OPERATOR_EVENT_KINDS,
    OperatorAuth,
    is_operator_kind,
)


def test_operator_event_kinds_includes_expected_set() -> None:
    expected = {
        "private_simulation",
        "self_world",
        "quarantine",
        "divergence",
        "snapshot_diff",
        "meta_proposal",
        "code_gen",
        "narrative",
        "research_finding",
        "subsystem_event",
    }
    assert expected <= set(OPERATOR_EVENT_KINDS)


def test_is_operator_kind_classifies_correctly() -> None:
    assert is_operator_kind("divergence")
    assert is_operator_kind("meta_proposal")
    assert not is_operator_kind("chat")
    assert not is_operator_kind("self_modification")
    assert not is_operator_kind("simulation")  # public sim, not the private one


def test_auth_without_token_accepts_any_supplied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DARWIN_OPERATOR_TOKEN", raising=False)
    auth = OperatorAuth()
    assert auth.verify(None)
    assert auth.verify("anything")
    assert not auth.is_configured()


def test_auth_with_token_requires_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DARWIN_OPERATOR_TOKEN", "secret")
    auth = OperatorAuth()
    assert auth.is_configured()
    assert auth.verify("secret")
    assert not auth.verify("wrong")
    assert not auth.verify(None)
    assert not auth.verify("")
