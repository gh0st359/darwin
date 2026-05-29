"""Vision modality: camera frames → online feature digest → causal substrate.

This adapter tries OpenCV first; if it isn't installed (or no device is
present), it falls back to inactive and emits nothing. The intended pipeline
is: capture a frame → quantize to a small grid of mean-intensity buckets →
embed that into the same self-trained CausalEmbeddingSpace via token sequence
(no pretrained vision weights). v9 wires the embedder; for v9.0 the adapter
ships the capture loop.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from darwin.types import Transition


def _try_cv2() -> Any:  # pragma: no cover - hardware-dependent
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


@dataclass
class VisionModalityAdapter:
    device_index: int = 0
    track: str = "public"
    grid: int = 4
    _cv2: Any = None
    _capture: Any = None
    _t: int = 0
    active: bool = False
    _last_digest: str = ""

    def __post_init__(self) -> None:  # pragma: no cover - hardware-dependent
        self._cv2 = _try_cv2()
        if self._cv2 is None:
            self.active = False
            return
        try:
            cap = self._cv2.VideoCapture(self.device_index)
            if not cap.isOpened():
                self.active = False
                return
            self._capture = cap
            self.active = True
        except Exception:
            self.active = False

    def _bucket_frame(self, frame: Any) -> list[int]:  # pragma: no cover
        h, w = frame.shape[:2]
        ys = max(1, h // self.grid)
        xs = max(1, w // self.grid)
        out: list[int] = []
        for gy in range(self.grid):
            for gx in range(self.grid):
                patch = frame[gy * ys: (gy + 1) * ys, gx * xs: (gx + 1) * xs]
                out.append(int(patch.mean()))
        return out

    def observe(self) -> list[Transition]:  # pragma: no cover - hardware-dependent
        if not self.active or self._capture is None:
            return []
        try:
            ok, frame = self._capture.read()
        except Exception:
            return []
        if not ok or frame is None:
            return []
        bucket = self._bucket_frame(frame)
        digest = hashlib.sha256(bytes(bucket)).hexdigest()
        if digest == self._last_digest:
            return []
        prior = self._last_digest
        self._last_digest = digest
        self._t += 1
        return [
            Transition(
                before={"frame_sha": prior},
                action="vision:frame",
                after={"frame_sha": digest, "buckets": bucket},
                reward=0.0,
                t=self._t,
                metadata={"track": self.track, "modality": "vision"},
            )
        ]

    def status(self) -> dict[str, Any]:
        return {
            "modality": "vision",
            "active": self.active,
            "device_index": self.device_index,
            "grid": self.grid,
            "track": self.track,
        }
