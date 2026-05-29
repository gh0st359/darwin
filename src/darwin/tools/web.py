"""WebTool — fetch URLs and extract text. No external dependencies.

Stdlib-only. ``urllib`` performs the fetch; an HTML parser strips tags and
collapses whitespace to a plain-text body. By default the tool refuses
``file://`` and ``ftp://`` URLs (only ``http``/``https`` are allowed) and
caps response size to keep one bad URL from filling the universe.
"""

from __future__ import annotations

import html
import re
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Any

from darwin.tools.base import Tool, ToolResult
from darwin.types import Action


_ALLOWED_SCHEMES = ("http", "https")
_MAX_FETCH_BYTES = 1_500_000      # 1.5 MiB
_DEFAULT_USER_AGENT = "DarwinBrain/1.0 (+https://github.com/gh0st359/darwin)"


class _TextExtractor(HTMLParser):
    """Minimal HTML → text. Drops <script> and <style> entirely."""

    def __init__(self) -> None:
        super().__init__()
        self._buf: list[str] = []
        self._skip_depth = 0
        self._skip_tags = {"script", "style", "noscript"}
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs) -> None:
        if tag in self._skip_tags:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag) -> None:
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self.title += data
        else:
            self._buf.append(data)

    def text(self) -> str:
        joined = "".join(self._buf)
        # Collapse runs of whitespace, including newlines, into single spaces;
        # preserve paragraph-like breaks as a single newline.
        joined = html.unescape(joined)
        # Split on blank-line-ish boundaries.
        chunks = re.split(r"\n\s*\n+", joined)
        out: list[str] = []
        for chunk in chunks:
            normalized = re.sub(r"\s+", " ", chunk).strip()
            if normalized:
                out.append(normalized)
        return "\n\n".join(out)


def _extract_text(body: bytes, encoding: str = "utf-8") -> tuple[str, str]:
    try:
        text = body.decode(encoding, errors="replace")
    except LookupError:
        text = body.decode("utf-8", errors="replace")
    parser = _TextExtractor()
    parser.feed(text)
    return parser.text(), parser.title.strip()


class WebTool(Tool):
    """Fetch a URL and return the extracted body text."""

    name = "web"
    description = "Fetch a URL (http/https) and return its text body."

    def __init__(
        self,
        *,
        timeout_seconds: float = 8.0,
        max_bytes: int = _MAX_FETCH_BYTES,
        user_agent: str = _DEFAULT_USER_AGENT,
        max_text_chars: int = 8_000,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.max_bytes = int(max_bytes)
        self.user_agent = user_agent
        self.max_text_chars = int(max_text_chars)

    def actions(self) -> list[Action]:
        return [
            Action("web_fetch", cost=0.0, description="fetch a URL and return its text"),
        ]

    def execute(self, input: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        url = str(input.get("url", "")).strip()
        if not url:
            return self._wrap(
                "web_fetch", started, False, "",
                error="empty url",
                input_payload=input,
            )
        scheme = urllib.parse.urlsplit(url).scheme.lower() if hasattr(urllib, "parse") else ""
        # Re-import explicitly to avoid an attribute issue on some envs.
        from urllib.parse import urlsplit

        scheme = urlsplit(url).scheme.lower()
        if scheme not in _ALLOWED_SCHEMES:
            return self._wrap(
                "web_fetch", started, False, "",
                error=f"scheme {scheme!r} not allowed (http/https only)",
                input_payload=input,
            )
        request = urllib.request.Request(
            url,
            headers={"User-Agent": self.user_agent, "Accept": "text/html, text/*;q=0.9, */*;q=0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                body = response.read(self.max_bytes + 1)
                truncated = len(body) > self.max_bytes
                body = body[: self.max_bytes]
                content_type = response.headers.get("Content-Type", "").lower()
        except urllib.error.URLError as exc:
            return self._wrap(
                "web_fetch", started, False, "",
                error=f"URLError: {exc}",
                input_payload=input,
            )
        except TimeoutError as exc:
            return self._wrap(
                "web_fetch", started, False, "",
                error=f"timeout after {self.timeout_seconds:.1f}s",
                input_payload=input,
                metadata={"timeout": True},
            )
        text, title = _extract_text(body, charset)
        if len(text) > self.max_text_chars:
            text = text[: self.max_text_chars] + "\n... [truncated]"
        return self._wrap(
            "web_fetch", started, True, text,
            input_payload=input,
            metadata={
                "title": title,
                "content_type": content_type,
                "bytes": len(body),
                "truncated_fetch": truncated,
            },
        )
