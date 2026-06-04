"""CooccurrenceWindow — bounded ring of recent tokens producing pairs.

For skip-gram training we don't want to hold the full corpus in memory.
Instead, ingest streams tokens through a bounded window; at every push
the window emits `(center, context, distance)` pairs for the latest
token paired against every other token within the configured radius.

The window is intentionally lossy. Once the buffer fills, the oldest
tokens roll out. This matches how the trainer is meant to consume the
stream: as a flowing source, not a transactional log.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass
class CoocPair:
    center: str
    context: str
    distance: int  # 1..window

    def __iter__(self) -> Iterator:
        return iter((self.center, self.context, self.distance))


class CooccurrenceWindow:
    """Sliding-window co-occurrence emitter."""

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        # Buffer holds the last `window` tokens *before* the current center,
        # so the freshly-pushed token is always the center of the pair.
        self._buf: deque[str] = deque(maxlen=self.window)

    def push(self, token: str) -> list[CoocPair]:
        """Add ``token`` and return pairs (token, context, distance)."""

        pairs: list[CoocPair] = []
        for distance, ctx in enumerate(reversed(self._buf), start=1):
            pairs.append(CoocPair(center=token, context=ctx, distance=distance))
        self._buf.append(token)
        return pairs

    def push_stream(self, tokens: Iterable[str]) -> Iterator[CoocPair]:
        for tok in tokens:
            for pair in self.push(tok):
                yield pair

    def reset(self) -> None:
        self._buf.clear()

    def size(self) -> int:
        return len(self._buf)


__all__ = ["CooccurrenceWindow", "CoocPair"]
