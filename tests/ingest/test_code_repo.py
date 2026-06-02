"""Tests for CodeRepoIngester."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from darwin.ingest.code_repo import CodeRepoIngester


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
    ):
        subprocess.run(["git", "-C", str(root), "config", key, value],
                       check=True, capture_output=True)
    (root / "main.py").write_text(
        "class Foo:\n"
        "    '''class doc'''\n"
        "    def bar(self):\n"
        "        '''method doc'''\n"
        "        return 1\n"
        "\n"
        "def standalone():\n"
        "    '''module-level function'''\n"
        "    return 2\n"
    )
    (root / "helper.js").write_text(
        "function greet(name) { return `hello ${name}`; }\n"
        "class Greeter {}\n"
    )
    subprocess.run(["git", "-C", str(root), "add", "."], check=True,
                   capture_output=True)
    commit = subprocess.run(
        ["git", "-C", str(root), "-c", "commit.gpgsign=false", "commit", "-m", "init"],
        capture_output=True,
    )
    if commit.returncode != 0:
        pytest.skip("git commit unavailable in test env")
    return root


def test_list_tracked_files_includes_committed_files(repo: Path) -> None:
    ingester = CodeRepoIngester()
    files = ingester.list_tracked_files(repo)
    names = {p.name for p in files}
    assert "main.py" in names
    assert "helper.js" in names


def test_ingest_repo_extracts_python_symbols(repo: Path) -> None:
    ingester = CodeRepoIngester()
    result = ingester.ingest_repo(repo)
    names = {s.name for s in result.symbols}
    assert "Foo" in names
    assert "bar" in names
    assert "standalone" in names


def test_ingest_repo_extracts_generic_js_symbols(repo: Path) -> None:
    ingester = CodeRepoIngester()
    result = ingester.ingest_repo(repo)
    js_symbols = [s for s in result.symbols if s.path.endswith(".js")]
    js_names = {s.name for s in js_symbols}
    assert "greet" in js_names
    assert "Greeter" in js_names


def test_ingest_repo_records_doctring_for_python_symbols(repo: Path) -> None:
    ingester = CodeRepoIngester()
    result = ingester.ingest_repo(repo)
    foo = next(s for s in result.symbols if s.name == "Foo" and s.kind == "class")
    assert "class doc" in foo.docstring


def test_ingest_repo_handles_missing_repo() -> None:
    ingester = CodeRepoIngester()
    result = ingester.ingest_repo("/nonexistent/path/to/repo")
    assert result.error
    assert result.symbols == []


def test_ingest_repo_files_scanned_count(repo: Path) -> None:
    ingester = CodeRepoIngester()
    result = ingester.ingest_repo(repo)
    assert result.files_scanned >= 2
