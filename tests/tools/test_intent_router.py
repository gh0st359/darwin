"""Tests for the chat→tool intent router."""

from __future__ import annotations

from darwin.tools.intent import IntentMatch, detect_intents


def test_no_intent_in_neutral_chat() -> None:
    assert detect_intents("Tell me about gravity.") == []


def test_empty_message_returns_no_intent() -> None:
    assert detect_intents("") == []


def test_fs_list_intent_detected() -> None:
    matches = detect_intents("Can you list the files in /workspace?")
    assert any(m.action == "fs_list" and m.input["path"] == "/workspace" for m in matches)


def test_fs_list_supports_what_is_in_phrasing() -> None:
    matches = detect_intents("What's in /tmp?")
    # 'whats in /tmp' should resolve to fs_list /tmp.
    assert any(m.action == "fs_list" for m in matches)


def test_bare_url_routes_to_web_fetch() -> None:
    matches = detect_intents("https://example.com")
    assert any(m.action == "web_fetch" and m.input["url"].startswith("https://") for m in matches)


def test_explicit_fetch_verb_with_url() -> None:
    matches = detect_intents("Please fetch https://example.com/x")
    assert any(m.action == "web_fetch" and m.input["url"] == "https://example.com/x" for m in matches)


def test_shell_command_intent() -> None:
    matches = detect_intents("run echo hello")
    assert any(m.action == "shell" and "echo hello" in m.input["command"] for m in matches)


def test_python_block_intent() -> None:
    matches = detect_intents(
        "run python:\n```python\nprint('hi')\n```"
    )
    assert any(m.action == "exec_python" and "print('hi')" in m.input["source"] for m in matches)


def test_git_status_intent() -> None:
    matches = detect_intents("What is in the current git repo?")
    assert any(m.action == "git_status" for m in matches)


def test_git_log_intent() -> None:
    matches = detect_intents("Show recent commits.")
    assert any(m.action == "git_log" for m in matches)


def test_select_sql_routes_to_db_query() -> None:
    matches = detect_intents("Query: SELECT * FROM notes")
    assert any(
        m.action == "db_query" and m.input["sql"].lower().startswith("select")
        for m in matches
    )


def test_non_select_sql_is_not_auto_routed() -> None:
    # We deliberately only auto-route SELECT/WITH. DROP/DELETE etc.
    # require explicit shell/exec to surface destructive intent.
    matches = detect_intents("Query: DROP TABLE notes")
    assert all(m.action != "db_query" for m in matches)


def test_intent_match_serializes() -> None:
    intent = IntentMatch(action="fs_list", input={"path": "."}, confidence=0.7, reason="r")
    record = intent.to_record()
    assert record["action"] == "fs_list"
    assert record["input"] == {"path": "."}


def test_multiple_intents_can_coexist() -> None:
    matches = detect_intents("Please fetch https://example.com and show recent commits.")
    actions = {m.action for m in matches}
    assert "web_fetch" in actions
    assert "git_log" in actions
