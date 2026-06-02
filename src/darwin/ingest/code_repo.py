"""CodeRepoIngester — walk a git repository and build a symbol table."""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class Symbol:
    """One named definition in source code."""

    qualified_name: str           # e.g. "src/x.py::Foo.bar"
    name: str
    kind: str                     # "function" / "class" / "method"
    path: str
    line: int = 0
    docstring: str = ""


@dataclass
class RepoIngestResult:
    repo_root: str
    symbols: list[Symbol] = field(default_factory=list)
    files_scanned: int = 0
    facts_emitted: int = 0
    error: str = ""

    def to_record(self) -> dict:
        return {
            "repo_root": self.repo_root,
            "symbol_count": len(self.symbols),
            "files_scanned": self.files_scanned,
            "facts_emitted": self.facts_emitted,
            "error": self.error[:200],
        }


def _python_symbols(path: Path, source: str) -> list[Symbol]:
    """Extract Python symbols via ``ast``."""

    symbols: list[Symbol] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return symbols
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            symbols.append(Symbol(
                qualified_name=f"{path}::{node.name}",
                name=node.name,
                kind="function",
                path=str(path),
                line=node.lineno,
                docstring=(ast.get_docstring(node) or "")[:300],
            ))
        elif isinstance(node, ast.AsyncFunctionDef):
            symbols.append(Symbol(
                qualified_name=f"{path}::{node.name}",
                name=node.name,
                kind="async_function",
                path=str(path),
                line=node.lineno,
                docstring=(ast.get_docstring(node) or "")[:300],
            ))
        elif isinstance(node, ast.ClassDef):
            symbols.append(Symbol(
                qualified_name=f"{path}::{node.name}",
                name=node.name,
                kind="class",
                path=str(path),
                line=node.lineno,
                docstring=(ast.get_docstring(node) or "")[:300],
            ))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(Symbol(
                        qualified_name=f"{path}::{node.name}.{item.name}",
                        name=item.name,
                        kind="method",
                        path=str(path),
                        line=item.lineno,
                        docstring=(ast.get_docstring(item) or "")[:300],
                    ))
    return symbols


# Lightweight regex fallback for non-Python files.
_GENERIC_FUNCTION_RX = re.compile(
    r"^\s*(?:fn|function|def|sub|method|public|private)\s+([A-Za-z_][A-Za-z0-9_]*)\s*[(]",
    re.MULTILINE,
)
_GENERIC_CLASS_RX = re.compile(
    r"^\s*(?:class|interface|struct|trait)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)


def _generic_symbols(path: Path, source: str) -> list[Symbol]:
    out: list[Symbol] = []
    for m in _GENERIC_FUNCTION_RX.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        out.append(Symbol(
            qualified_name=f"{path}::{m.group(1)}",
            name=m.group(1),
            kind="function",
            path=str(path),
            line=line_no,
        ))
    for m in _GENERIC_CLASS_RX.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        out.append(Symbol(
            qualified_name=f"{path}::{m.group(1)}",
            name=m.group(1),
            kind="class",
            path=str(path),
            line=line_no,
        ))
    return out


class CodeRepoIngester:
    """Walk a git repo, extract a symbol table, optionally emit facts."""

    def __init__(self) -> None:
        self.repos_scanned = 0
        self.symbols_total = 0

    def list_tracked_files(self, repo_root: str | Path) -> list[Path]:
        """Return git-tracked files via ``git ls-files``. Empty on failure."""

        root = Path(repo_root)
        try:
            proc = subprocess.run(
                ["git", "-C", str(root), "ls-files"],
                capture_output=True, timeout=20, check=False,
            )
        except (subprocess.SubprocessError, OSError):
            return []
        if proc.returncode != 0:
            return []
        out: list[Path] = []
        for line in proc.stdout.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if not line:
                continue
            candidate = root / line
            if candidate.is_file():
                out.append(candidate)
        return out

    def ingest_repo(self, repo_root: str | Path) -> RepoIngestResult:
        root = Path(repo_root)
        result = RepoIngestResult(repo_root=str(root))
        files = self.list_tracked_files(root)
        if not files:
            result.error = "no files tracked by git"
            return result
        for path in files:
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            result.files_scanned += 1
            if path.suffix == ".py":
                result.symbols.extend(_python_symbols(path, source))
            elif path.suffix in (".js", ".ts", ".java", ".go", ".rs", ".c",
                                  ".cpp", ".rb", ".php", ".kt", ".scala"):
                result.symbols.extend(_generic_symbols(path, source))
        self.repos_scanned += 1
        self.symbols_total += len(result.symbols)
        return result


__all__ = ["CodeRepoIngester", "RepoIngestResult", "Symbol"]
