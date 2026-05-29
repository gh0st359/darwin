"""Tests for FilesystemTool sandbox semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from darwin.tools.base import SandboxEscape, resolve_sandboxed
from darwin.tools.filesystem import FilesystemTool


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    fs = FilesystemTool(tmp_path)
    write = fs.execute({"action": "fs_write", "path": "notes/hello.txt", "content": "hi"})
    assert write.success
    read = fs.execute({"action": "fs_read", "path": "notes/hello.txt"})
    assert read.success
    assert read.output == "hi"


def test_list_returns_directory_entries(tmp_path: Path) -> None:
    fs = FilesystemTool(tmp_path)
    fs.execute({"action": "fs_write", "path": "a.txt", "content": "1"})
    fs.execute({"action": "fs_write", "path": "subdir/b.txt", "content": "2"})
    listing = fs.execute({"action": "fs_list", "path": "."})
    assert listing.success
    assert "f a.txt" in listing.output
    assert "d subdir" in listing.output


def test_remove_deletes_file(tmp_path: Path) -> None:
    fs = FilesystemTool(tmp_path)
    fs.execute({"action": "fs_write", "path": "doomed.txt", "content": "x"})
    remove = fs.execute({"action": "fs_remove", "path": "doomed.txt"})
    assert remove.success
    assert not (tmp_path / "doomed.txt").exists()


def test_recursive_directory_removal_is_refused(tmp_path: Path) -> None:
    fs = FilesystemTool(tmp_path)
    fs.execute({"action": "fs_write", "path": "subdir/x.txt", "content": "x"})
    remove = fs.execute({"action": "fs_remove", "path": "subdir"})
    assert not remove.success
    assert "recursive" in remove.error.lower()


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    fs = FilesystemTool(tmp_path)
    result = fs.execute({"action": "fs_read", "path": "../../../etc/passwd"})
    assert not result.success
    assert "sandbox escape" in result.error.lower()


def test_absolute_path_outside_sandbox_is_rejected(tmp_path: Path) -> None:
    fs = FilesystemTool(tmp_path)
    result = fs.execute({"action": "fs_read", "path": "/etc/passwd"})
    assert not result.success
    assert "sandbox escape" in result.error.lower()


def test_read_size_limit_enforced(tmp_path: Path) -> None:
    fs = FilesystemTool(tmp_path, max_read_bytes=8)
    fs.execute({"action": "fs_write", "path": "big.txt", "content": "x" * 100})
    read = fs.execute({"action": "fs_read", "path": "big.txt"})
    assert not read.success
    assert "too large" in read.error.lower()


def test_write_size_limit_enforced(tmp_path: Path) -> None:
    fs = FilesystemTool(tmp_path, max_write_bytes=8)
    write = fs.execute({"action": "fs_write", "path": "big.txt", "content": "x" * 100})
    assert not write.success
    assert "too large" in write.error.lower()


def test_stat_reports_size_and_kind(tmp_path: Path) -> None:
    fs = FilesystemTool(tmp_path)
    fs.execute({"action": "fs_write", "path": "f.txt", "content": "ABC"})
    stat = fs.execute({"action": "fs_stat", "path": "f.txt"})
    assert stat.success
    assert "size=3" in stat.output
    assert stat.metadata["kind"] == "file"


def test_resolve_sandboxed_helper_raises_on_escape(tmp_path: Path) -> None:
    with pytest.raises(SandboxEscape):
        resolve_sandboxed(tmp_path, "../escape")
    with pytest.raises(SandboxEscape):
        resolve_sandboxed(tmp_path, "/etc/passwd")


def test_actions_lists_every_fs_operation(tmp_path: Path) -> None:
    fs = FilesystemTool(tmp_path)
    names = {a.name for a in fs.actions()}
    assert {"fs_read", "fs_write", "fs_list", "fs_remove", "fs_stat"} <= names


def test_unknown_action_returns_failed_result(tmp_path: Path) -> None:
    fs = FilesystemTool(tmp_path)
    result = fs.execute({"action": "fs_eat_homework", "path": "."})
    assert not result.success
    assert "unknown" in result.error.lower()
