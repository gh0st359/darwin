#!/usr/bin/env python3
"""Extract Mermaid blocks from Markdown, render SVGs, and embed images."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS = ROOT / "docs" / "diagrams"
MMDC = ["npx", "-y", "@mermaid-js/mermaid-cli@11.4.0", "-i"]
PATTERN = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)
MARKDOWN_ROOTS = (ROOT / "README.md", ROOT / "docs", ROOT / "wiki")


def slugify(path: Path, index: int) -> str:
    rel = path.relative_to(ROOT)
    parts = [part.lower().replace("_", "-") for part in rel.parts[:-1]]
    parts.append(rel.stem.lower().replace("_", "-"))
    return f"{'-'.join(parts)}-{index:02d}"


def render_mmd(mmd_path: Path, svg_path: Path) -> None:
    subprocess.run(
        [*MMDC, str(mmd_path), "-o", str(svg_path)],
        check=True,
        cwd=ROOT,
    )


def process_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    matches = list(PATTERN.finditer(text))
    if not matches:
        return 0

    updated = text
    offset = 0
    for index, match in enumerate(matches, start=1):
        source = match.group(1).strip() + "\n"
        slug = slugify(path.relative_to(ROOT), index)
        mmd_path = DIAGRAMS / f"{slug}.mmd"
        svg_path = DIAGRAMS / f"{slug}.svg"
        mmd_path.write_text(source, encoding="utf-8")
        render_mmd(mmd_path, svg_path)

        rel_svg = os.path.relpath(svg_path, path.parent).replace(os.sep, "/")
        title = slug.rsplit("-", 1)[0].replace("-", " ").title()
        replacement = f"![{title}]({rel_svg})"
        start = match.start() + offset
        end = match.end() + offset
        updated = updated[:start] + replacement + updated[end:]
        offset += len(replacement) - (match.end() - match.start())

    path.write_text(updated, encoding="utf-8")
    return len(matches)


def main() -> int:
    DIAGRAMS.mkdir(parents=True, exist_ok=True)
    total = 0
    paths: list[Path] = []
    for root in MARKDOWN_ROOTS:
        if root.is_file():
            paths.append(root)
        else:
            paths.extend(sorted(root.rglob("*.md")))

    for path in paths:
        count = process_file(path)
        if count:
            print(f"updated {path}: {count} diagram(s)")
            total += count

    print(f"rendered {total} diagram(s) into {DIAGRAMS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
