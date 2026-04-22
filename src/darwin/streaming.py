from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from typing import TextIO


@dataclass
class StreamingSpeaker:
    enabled: bool = True
    delay: float = 0.012

    def write(self, text: str, stream: TextIO | None = None) -> None:
        target = stream or sys.stdout
        if not self.enabled:
            target.write(text + "\n")
            target.flush()
            return

        for chunk in self._chunks(text):
            target.write(chunk)
            target.flush()
            if chunk.strip():
                time.sleep(self.delay)
        target.write("\n")
        target.flush()

    def _chunks(self, text: str) -> list[str]:
        pieces = re.findall(r"\S+\s*", text)
        if not pieces:
            return [text]
        return pieces

