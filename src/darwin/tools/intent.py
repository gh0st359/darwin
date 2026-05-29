"""IntentRouter — translate user chat into tool invocations when natural.

When the operator says "list the files in /workspace" or "run echo hello" or
"fetch https://example.com", a frontier system should *use the appropriate
tool* and weave the result into its reply — not refuse on the grounds that
it's "just a language model". The IntentRouter is the bridge.

It is *rule-based and conservative*. False negatives are preferred over
false positives — confidently mis-routing a question into a tool call is
worse than missing one. The router returns a structured
:class:`IntentMatch` (or None) describing which tool to invoke, with what
input. The runtime decides whether to execute it.

This is a routing layer, not a planning layer. For multi-step tool
sequences, the AutonomousRunner remains the right abstraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IntentMatch:
    """A single recognized tool intent."""

    action: str
    input: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    reason: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "input": dict(self.input),
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
        }


# Regexes. Each one captures whatever subgroups the action needs.

_URL_RX = re.compile(
    r"\b(?:fetch|get|open|read|grab|download|retrieve|look at)\s+"
    r"(?P<url>https?://[^\s'\"]+)",
    re.IGNORECASE,
)
_URL_BARE_RX = re.compile(
    r"^\s*(?P<url>https?://[^\s'\"]+)\s*\??\s*$",
    re.IGNORECASE,
)
_FS_LIST_RX = re.compile(
    r"\b(?:list|show|ls)\s+(?:the\s+)?"
    r"(?:files?|contents|directory)\s+"
    # If a preposition is present, it must be followed by whitespace so
    # it can't be captured as the path itself.
    r"(?:(?:in|of|at|under)\s+)?"
    # Path may include "." (the current dir is itself a valid listing
    # target). Trailing punctuation is stripped by _strip_quotes.
    r"(?P<path>['\"]?[^\s'\"?,]+['\"]?)",
    re.IGNORECASE,
)
# Short "what's in <path>?" pattern that doesn't require the
# files/contents/directory keyword.
_FS_LIST_SHORT_RX = re.compile(
    r"\bwhat(?:'s|s| is|s in)\s+(?:in|at|under)\s+"
    r"(?P<path>['\"]?[^\s'\"?.,]+['\"]?)\s*\??",
    re.IGNORECASE,
)
_FS_READ_RX = re.compile(
    r"\b(?:read|cat|show|open|display|print|dump)\s+(?:the\s+)?"
    r"(?:file|contents of)?\s*"
    r"(?P<path>['\"]?[A-Za-z0-9_./\-]+['\"]?)",
    re.IGNORECASE,
)
_SHELL_RX = re.compile(
    r"^\s*(?:run|execute|exec|sh)\s+(?:the\s+command\s+)?"
    r"['\"`]?(?P<command>[^'\"`]+)['\"`]?\s*\??$",
    re.IGNORECASE,
)
_PYTHON_RX = re.compile(
    r"\b(?:run|execute|exec|eval)\s+(?:this\s+)?python(?:\s+code)?:?\s*\n?"
    r"```?(?:python)?\n?(?P<source>.+?)```?",
    re.IGNORECASE | re.DOTALL,
)
_GIT_STATUS_RX = re.compile(
    r"\bgit\s+status\b"
    r"|\bwhat(?:'s| is)?\s+(?:in|the)\s+(?:the\s+)?(?:current\s+)?(?:git\s+)?repo(?:sitory)?",
    re.IGNORECASE,
)
_GIT_LOG_RX = re.compile(
    r"\b(?:git\s+log|show\s+(?:recent\s+|the\s+)?commits|what(?:'s| is)\s+the\s+recent\s+history)",
    re.IGNORECASE,
)
_SQL_RX = re.compile(
    r"\b(?:query|select|run\s+sql):?\s*(?P<sql>.+?)$",
    re.IGNORECASE,
)


def _strip_quotes(value: str) -> str:
    value = value.strip().strip("'").strip('"').strip("`")
    return value.rstrip(".,!?")


def detect_intents(message: str) -> list[IntentMatch]:
    """Scan a chat message for tool-routable intents. Returns 0+ matches."""

    if not message:
        return []
    matches: list[IntentMatch] = []

    # 1. Filesystem list (canonical form + short "what's in X?" form).
    m = _FS_LIST_RX.search(message)
    if m:
        path = _strip_quotes(m.group("path"))
        matches.append(IntentMatch(
            action="fs_list",
            input={"path": path or "."},
            confidence=0.7,
            reason=f"user appears to want a directory listing of {path!r}",
        ))
    else:
        m = _FS_LIST_SHORT_RX.search(message)
        if m:
            path = _strip_quotes(m.group("path"))
            # Short form only fires when the captured "path" looks like a
            # real path (contains a slash or a dot) — never on bare
            # English words like "the" or "current".
            if "/" in path or "." in path:
                matches.append(IntentMatch(
                    action="fs_list",
                    input={"path": path},
                    confidence=0.65,
                    reason=f"user appears to want a directory listing of {path!r}",
                ))

    # 2. Web fetch (explicit verb + URL).
    m = _URL_RX.search(message)
    if m:
        matches.append(IntentMatch(
            action="web_fetch",
            input={"url": m.group("url")},
            confidence=0.85,
            reason="user named a URL and a fetch verb",
        ))
    else:
        # Bare URL as the entire message also counts.
        m = _URL_BARE_RX.match(message)
        if m:
            matches.append(IntentMatch(
                action="web_fetch",
                input={"url": m.group("url")},
                confidence=0.75,
                reason="message is just a URL",
            ))

    # 3. Shell.
    m = _SHELL_RX.match(message)
    if m:
        matches.append(IntentMatch(
            action="shell",
            input={"command": m.group("command").strip()},
            confidence=0.75,
            reason="user explicitly asked to run a shell command",
        ))

    # 4. Python.
    m = _PYTHON_RX.search(message)
    if m:
        matches.append(IntentMatch(
            action="exec_python",
            input={"source": m.group("source").strip()},
            confidence=0.8,
            reason="user supplied a Python code block to execute",
        ))

    # 5. Git status.
    if _GIT_STATUS_RX.search(message) and not _SHELL_RX.match(message):
        matches.append(IntentMatch(
            action="git_status",
            input={},
            confidence=0.7,
            reason="user appears to want git status",
        ))

    # 6. Git log.
    if _GIT_LOG_RX.search(message) and not _SHELL_RX.match(message):
        matches.append(IntentMatch(
            action="git_log",
            input={"args": ["--oneline", "-10"]},
            confidence=0.65,
            reason="user appears to want recent git history",
        ))

    # 7. Filesystem read (lower-confidence; check after list because "show"
    # is overloaded).
    if not any(m.action == "fs_list" for m in matches):
        m = _FS_READ_RX.search(message)
        if m:
            path = _strip_quotes(m.group("path"))
            # Heuristic guard: only fire if it looks like a real path.
            if "." in path or "/" in path:
                matches.append(IntentMatch(
                    action="fs_read",
                    input={"path": path},
                    confidence=0.6,
                    reason=f"user appears to want the contents of {path!r}",
                ))

    # 8. SQL.
    m = _SQL_RX.search(message)
    if m and m.group("sql").lower().lstrip().startswith(("select", "with")):
        matches.append(IntentMatch(
            action="db_query",
            input={"sql": m.group("sql").strip()},
            confidence=0.6,
            reason="user supplied a SQL query",
        ))

    return matches


__all__ = ["IntentMatch", "detect_intents"]
