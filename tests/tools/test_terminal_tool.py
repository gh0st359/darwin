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
    # Strict mode rejects shell metacharacters first; even without them
    # the binary "rm" is on the denylist.
    err = result.error.lower()
    assert "denylist" in err or "metacharacters" in err or "not in allowlist" in err


def test_shell_metacharacters_rejected(tmp_path: Path) -> None:
    term = TerminalTool(tmp_path)
    for command in (
        "echo a; echo b",
        "echo a && echo b",
        "echo a | cat",
        "echo $(whoami)",
        "echo a > /tmp/x",
    ):
        result = term.execute({"command": command})
        assert not result.success
        assert "metacharacters" in result.error.lower()


def test_denylist_blocks_rm_even_without_meta(tmp_path: Path) -> None:
    term = TerminalTool(tmp_path)
    result = term.execute({"command": "rm somefile"})
    assert not result.success
    assert "denylist" in result.error.lower()


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
    term = TerminalTool(tmp_path, timeout_seconds=0.5, extra_allowed=("sleep",))
    result = term.execute({"command": "sleep 5"})
    assert not result.success
    assert "timeout" in result.error.lower()


def test_output_truncated_to_configured_size(tmp_path: Path) -> None:
    term = TerminalTool(tmp_path, max_output_bytes=64)
    # Compose via a real argv array shlex can parse cleanly.
    result = term.execute(
        {"command": "python3 -c \"print('X'*500)\""}
    )
    assert result.success
    assert len(result.output) <= 64
