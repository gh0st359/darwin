"""DocumentIngester — plain text and HTML."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

from darwin.ingest.nl_parser import Fact, NLParser


class _HTMLBodyExtractor(HTMLParser):
    """Minimal HTML → text. Drops <script>, <style>, <noscript>."""

    _SKIP_TAGS = {"script", "style", "noscript", "head"}

    def __init__(self) -> None:
        super().__init__()
        self._buf: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data) -> None:
        if self._skip_depth > 0:
            return
        self._buf.append(data)

    def text(self) -> str:
        joined = "".join(self._buf)
        # Normalise whitespace; preserve paragraph breaks.
        chunks = re.split(r"\n\s*\n+", joined)
        out: list[str] = []
        for chunk in chunks:
            normalised = re.sub(r"\s+", " ", chunk).strip()
            if normalised:
                out.append(normalised)
        return "\n\n".join(out)


@dataclass
class IngestResult:
    """Outcome of one ingest operation."""

    source: str
    facts: list[Fact] = field(default_factory=list)
    sentences_seen: int = 0
    duration_ms: float = 0.0
    error: str = ""

    def to_record(self) -> dict:
        return {
            "source": self.source,
            "fact_count": len(self.facts),
            "sentences_seen": self.sentences_seen,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error[:200],
        }


class DocumentIngester:
    """Ingest plain text or HTML into Facts via the NLParser."""

    def __init__(self, parser: NLParser | None = None) -> None:
        self.parser = parser or NLParser()
        self.total_facts = 0
        self.total_sources = 0

    def ingest_text(self, text: str, *, source: str = "text") -> IngestResult:
        import time as _time

        started = _time.perf_counter()
        result = IngestResult(source=source)
        try:
            facts = self.parser.parse(text)
            result.facts = facts
            result.sentences_seen = len(facts)
            self.total_facts += len(facts)
            self.total_sources += 1
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        result.duration_ms = max(0.0, (_time.perf_counter() - started) * 1000.0)
        return result

    def ingest_html(self, html_text: str, *, source: str = "html") -> IngestResult:
        extractor = _HTMLBodyExtractor()
        try:
            extractor.feed(html_text or "")
        except Exception:
            return IngestResult(source=source, error="malformed HTML")
        plain = extractor.text()
        return self.ingest_text(plain, source=source)

    def ingest_file(self, path: str | Path) -> IngestResult:
        target = Path(path)
        if not target.exists():
            return IngestResult(source=str(target), error="file not found")
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return IngestResult(source=str(target), error=str(exc))
        if target.suffix.lower() in (".html", ".htm"):
            return self.ingest_html(content, source=str(target))
        return self.ingest_text(content, source=str(target))


__all__ = ["DocumentIngester", "IngestResult"]
