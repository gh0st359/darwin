"""Tests for GitTool against a freshly-initialized sandbox repo."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from darwin.tools.git import GitTool


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(root)], check=True,
                   capture_output=True)
    for key, value in (
        ("user.email", "test@example.com"),
        ("user.name", "test"),
        ("commit.gpgsign", "false"),
        ("tag.gpgsign", "false"),
    ):
        subprocess.run(["git", "-C", str(root), "config", key, value],
                       check=True, capture_output=True)
    (root / "README.md").write_text("# repo\n")
    subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True,
                   capture_output=True)
    commit = subprocess.run(
        ["git", "-C", str(root), "-c", "commit.gpgsign=false", "commit", "-m", "init"],
        capture_output=True,
    )
    if commit.returncode != 0:
        pytest.skip(
            f"git commit unavailable in test env: rc={commit.returncode} "
            f"stderr={commit.stderr.decode('utf-8', 'replace')[:200]}"
        )
    return root


def test_status_clean_repo_succeeds(repo: Path) -> None:
    tool = GitTool(repo)
    result = tool.execute({"action": "git_status"})
    assert result.success
    assert "clean" in result.output.lower() or "nothing to commit" in result.output.lower()


def test_log_oneline(repo: Path) -> None:
    tool = GitTool(repo)
    result = tool.execute({"action": "git_log", "args": ["--oneline"]})
    assert result.success
    assert "init" in result.output


def test_diff_after_modification(repo: Path) -> None:
    (repo / "README.md").write_text("# modified\n")
    tool = GitTool(repo)
    result = tool.execute({"action": "git_diff"})
    assert result.success
    assert "modified" in result.output


def test_write_subcommand_refused(repo: Path) -> None:
    tool = GitTool(repo)
    result = tool.execute({"action": "git_commit", "args": ["-m", "no"]})
    assert not result.success
    assert "read-only allowlist" in result.error.lower()


def test_push_refused(repo: Path) -> None:
    tool = GitTool(repo)
    result = tool.execute({"action": "git_push"})
    assert not result.success
    assert "read-only allowlist" in result.error.lower()


def test_non_string_args_rejected(repo: Path) -> None:
    tool = GitTool(repo)
    result = tool.execute({"action": "git_log", "args": [123]})
    assert not result.success
    assert "non-string" in result.error.lower()
