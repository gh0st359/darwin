"""Audio modality: microphone capture → spectral digest → causal substrate.

Tries sounddevice first; if it isn't available (no input device, no library
installed), the adapter falls back to inactive and emits nothing. The
intended pipeline is: short audio frame → log-mel-style band power vector →
embed via the self-trained CausalEmbeddingSpace as a token sequence. No
pretrained audio weights are imported.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from darwin.types import Transition


def _try_sd() -> Any:  # pragma: no cover - hardware-dependent
    try:
        import sounddevice as sd  # type: ignore

        return sd
    except Exception:
        return None


@dataclass
class AudioModalityAdapter:
    sample_rate: int = 16000
    duration_seconds: float = 0.5
    bands: int = 8
    track: str = "public"
    _sd: Any = None
    _t: int = 0
    active: bool = False
    _last_digest: str = ""

    def __post_init__(self) -> None:  # pragma: no cover - hardware-dependent
        self._sd = _try_sd()
        self.active = self._sd is not None

    def _band_powers(self, samples: Any) -> list[int]:  # pragma: no cover
        n = len(samples)
        if n == 0:
            return [0] * self.bands
        step = max(1, n // self.bands)
        out: list[int] = []
        for b in range(self.bands):
            slice_ = samples[b * step: (b + 1) * step]
            power = int(sum(abs(float(x)) for x in slice_))
            out.append(power)
        return out

    def observe(self) -> list[Transition]:  # pragma: no cover - hardware-dependent
        if not self.active or self._sd is None:
            return []
        try:
            frames = int(self.sample_rate * self.duration_seconds)
            samples = self._sd.rec(frames, samplerate=self.sample_rate, channels=1)
            self._sd.wait()
        except Exception:
            return []
        flat = list(samples.flatten()) if hasattr(samples, "flatten") else list(samples)
        powers = self._band_powers(flat)
        digest = hashlib.sha256(",".join(str(p) for p in powers).encode()).hexdigest()
        if digest == self._last_digest:
            return []
        prior = self._last_digest
        self._last_digest = digest
        self._t += 1
        return [
            Transition(
                before={"audio_sha": prior},
                action="audio:frame",
                after={"audio_sha": digest, "powers": powers},
                reward=0.0,
                t=self._t,
                metadata={"track": self.track, "modality": "audio"},
            )
        ]

    def status(self) -> dict[str, Any]:
        return {
            "modality": "audio",
            "active": self.active,
            "sample_rate": self.sample_rate,
            "bands": self.bands,
            "track": self.track,
        }
