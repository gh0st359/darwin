from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LiveResearchConfig:
    enabled: bool = False
    max_bytes: int = 250_000
    trust_floor: float = 0.5


class LiveResearcher:
    """Dormant live-web research interface.

    v4 exposes this capability shape now, but keeps it disabled by
    default so live web content cannot silently enter Darwin's belief
    stream before provenance and poisoning controls are enabled.
    """

    def __init__(self, config: LiveResearchConfig | None = None) -> None:
        self.config = config or LiveResearchConfig()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "max_bytes": self.config.max_bytes,
            "trust_floor": self.config.trust_floor,
            "mode": "dormant" if not self.config.enabled else "enabled",
        }

    def fetch(self, url: str) -> dict[str, Any]:
        if not self.config.enabled:
            raise PermissionError("live research is disabled by default")
        raise NotImplementedError("live research activation requires explicit future enablement")
