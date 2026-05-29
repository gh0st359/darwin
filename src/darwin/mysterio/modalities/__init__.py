"""Multi-modal ingest: vision, audio, filesystem, web.

Each adapter turns a stream of external observations into transitions on a
chosen track. Default track is ``public`` (a real filesystem change or a real
camera frame is grounded experience); a v7 caller can route any of them onto
a private track instead, exactly the same shape.

The adapters degrade gracefully: if a hardware/library backend isn't
available (no camera, no audio device, network blocked), the adapter still
constructs and reports inactive. This lets the modality roster live in the
default supervisor without forcing every dev environment to have a webcam.
"""

from darwin.mysterio.modalities.code import CodeModalityAdapter
from darwin.mysterio.modalities.web import WebModalityAdapter

# Vision and audio import lazily so a missing OpenCV/sounddevice install
# doesn't break the rest of the package on import.
try:  # pragma: no cover - optional
    from darwin.mysterio.modalities.vision import VisionModalityAdapter
except Exception:  # pragma: no cover
    VisionModalityAdapter = None  # type: ignore[assignment]
try:  # pragma: no cover
    from darwin.mysterio.modalities.audio import AudioModalityAdapter
except Exception:  # pragma: no cover
    AudioModalityAdapter = None  # type: ignore[assignment]


__all__ = [
    "CodeModalityAdapter",
    "WebModalityAdapter",
    "VisionModalityAdapter",
    "AudioModalityAdapter",
]
