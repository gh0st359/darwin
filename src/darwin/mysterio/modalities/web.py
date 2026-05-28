"""Web modality: fetched HTTP content as observable state.

Each call to :meth:`fetch` produces a transition whose ``after`` is a digest
of the response body (length, sha, content-type, status). The adapter does
not require an outbound network; if the fetch fails it returns an inactive
status and emits nothing. Designed to be polled by a subsystem that hands it
a small list of URLs from the operator console.
"""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from darwin.types import Transition


@dataclass
class WebFetchResult:
    url: str
    ok: bool
    status: int = 0
    content_type: str = ""
    length: int = 0
    sha256: str = ""
    error: str = ""


@dataclass
class WebModalityAdapter:
    track: str = "public"
    timeout: float = 4.0
    _seen_hashes: dict[str, str] = field(default_factory=dict)
    _t: int = 0
    active: bool = True

    def fetch(self, url: str) -> WebFetchResult:
        if not self.active:
            return WebFetchResult(url=url, ok=False, error="adapter inactive")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "darwin/mysterio"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
                status = resp.getcode()
                content_type = resp.headers.get("Content-Type", "")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return WebFetchResult(url=url, ok=False, error=repr(exc))
        sha = hashlib.sha256(body).hexdigest()
        return WebFetchResult(
            url=url, ok=True, status=int(status),
            content_type=content_type, length=len(body), sha256=sha,
        )

    def observe(self, urls: list[str]) -> list[Transition]:
        transitions: list[Transition] = []
        for url in urls:
            result = self.fetch(url)
            prior_sha = self._seen_hashes.get(url, "")
            self._t += 1
            transitions.append(
                Transition(
                    before={"url": url, "sha": prior_sha},
                    action="web:fetched" if result.ok else "web:failed",
                    after={
                        "url": url,
                        "sha": result.sha256,
                        "length": result.length,
                        "status": result.status,
                        "content_type": result.content_type,
                        "error": result.error,
                    },
                    reward=0.0,
                    t=self._t,
                    metadata={"track": self.track, "modality": "web"},
                )
            )
            if result.ok:
                self._seen_hashes[url] = result.sha256
        return transitions

    def status(self) -> dict[str, Any]:
        return {
            "modality": "web",
            "active": self.active,
            "tracked_urls": len(self._seen_hashes),
            "track": self.track,
        }
