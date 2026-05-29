"""Tests for TerminalTool."""

from __future__ import annotations

from pathlib import Path

from darwin.tools.terminal import TerminalTool


def test_simple_echo_succeeds(tmp_path: Path) -> None:
    term = TerminalTool(tmp_path)
    result = term.execute({"command": "echo hello"})
    assert result.success
    assert "hello" in result.output


def test_command_runs_in_sandbox_cwd(tmp_path: Path) -> None:
    term = TerminalTool(tmp_path)
    result = term.execute({"command": "pwd"})
    assert result.success
    assert str(tmp_path.resolve()) in result.output


def test_failing_command_returns_unsuccess(tmp_path: Path) -> None:
    term = TerminalTool(tmp_path)
    result = term.execute({"command": "false"})
    assert not result.success
    assert result.metadata["returncode"] != 0


def test_deny_list_rejects_dangerous_pattern(tmp_path: Path) -> None:
    term = TerminalTool(tmp_path)
    result = term.execute({"command": "rm -rf /"})
    assert not result.success
    assert "deny-list" in result.error.lower()


def test_empty_command_is_rejected(tmp_path: Path) -> None:
    term = TerminalTool(tmp_path)
    result = term.execute({"command": ""})
    assert not result.success
    assert "empty" in result.error.lower()


def test_allowed_command_list_enforces_first_token(tmp_path: Path) -> None:
    term = TerminalTool(tmp_path, allowed_commands=["echo"])
    ok = term.execute({"command": "echo allowed"})
    nope = term.execute({"command": "ls"})
    assert ok.success
    assert not nope.success
    assert "allowlist" in nope.error.lower()


def test_timeout_returns_failed_result(tmp_path: Path) -> None:
    term = TerminalTool(tmp_path, timeout_seconds=0.5)
    result = term.execute({"command": "sleep 5"})
    assert not result.success
    assert "timeout" in result.error.lower()


def test_output_truncated_to_configured_size(tmp_path: Path) -> None:
    term = TerminalTool(tmp_path, max_output_bytes=64)
    result = term.execute(
        {"command": "python3 -c 'print(\"X\" * 500)'"}
    )
    assert result.success
    assert len(result.output) <= 64
